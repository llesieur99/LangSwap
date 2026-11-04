"""Gradio demo: run the langswap video translation pipeline locally (no S3).

Launch with:
    python gradio_demo.py

Or with custom options:
    python gradio_demo.py --port 7860 --share

Uses LocalOnlyFileRepository so no AWS credentials are needed.  All
intermediate artifacts and the final dubbed video are written under
``data/<public_id>/`` relative to the working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import tempfile
import os
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

# Load .env (HF_TOKEN, ELEVEN_API_KEY, …) before anything else reads the
# environment.
load_dotenv()

from langswap import backends, editor
from langswap.file_repository import LocalOnlyFileRepository
from langswap.pipeline_models.models import TranslationPipelineConfig
from langswap.translation_pipeline import VideoTranslationPipeline
from langswap.youtube_source import download_existing_french, download_original

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("langswap.gradio")

BASE_DIR = os.environ.get("LANGSWAP_DATA_DIR", "data")
OUTPUT_DIR = os.environ.get("LANGSWAP_OUTPUT_DIR", "output")

LANGUAGES = ["english", "russian", "spanish", "french", "german", "italian",
             "portuguese", "polish", "dutch", "turkish", "arabic", "hindi",
             "japanese", "korean", "chinese"]

# Backend choices come from langswap/backends.json (single source of truth).
TTS_ENGINES = backends.options("tts")
ASR_BACKENDS = backends.options("asr")
TRANSLATION_BACKENDS = backends.options("translation")
DUBBING_ALGOS = backends.options("dubbing")


def _public_id(video_path: str) -> str:
    """Derive a stable ID from the input path so caches survive re-runs."""
    return hashlib.md5(str(video_path).encode("utf-8")).hexdigest()[:12]


def _resolve_device(requested: str) -> str:
    """Map a user-friendly device label to what the pipeline expects."""
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _export_results(video_path, source_srt, target_srt, output_dir, label, target_language):
    """Copy completed artifacts to one chosen folder, preserving previous runs."""
    folder = _resolve_output_dir(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")[:64] or "video"
    safe_lang = re.sub(r"[^a-z0-9_-]+", "-", target_language.lower()).strip("-")[:16]
    if not safe_lang:
        raise ValueError("Choose a target language.")
    base = f"LangSwap-{safe_label}.{safe_lang}"
    suffix = 1
    while any((folder / f"{base}{'' if suffix == 1 else '-' + str(suffix)}{ext}").exists()
              for ext in (".mp4", ".source.srt", ".translated.srt")):
        suffix += 1
    base += "" if suffix == 1 else f"-{suffix}"
    exports = []
    for src, ext in ((video_path, ".mp4"), (source_srt, ".source.srt"),
                     (target_srt, ".translated.srt")):
        destination = folder / f"{base}{ext}"
        with tempfile.NamedTemporaryFile(dir=folder, suffix=ext, delete=False) as temp:
            temporary = Path(temp.name)
        try:
            shutil.copyfile(src, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        exports.append(str(destination))
    return tuple(exports)


def _resolve_output_dir(output_dir):
    raw = str(output_dir or OUTPUT_DIR)
    if os.name != "nt" and ":" in raw:
        raise ValueError("Choose a folder on the machine running LangSwap.")
    folder = Path(raw).expanduser().resolve()
    if os.name == "nt":
        system32 = Path(os.environ.get("WINDIR", "C:\\Windows"), "System32").resolve()
        if folder == system32 or system32 in folder.parents:
            raise ValueError("The Windows system directory cannot be used for video output.")
    return folder


def translate_video(
    video_file: str,
    source_language: str,
    target_language: str,
    tts_engine: str,
    asr_backend: str,
    translation_backend: str,
    dubbing_algo: str,
    device: str,
    skip_diarization: bool,
    eleven_api_token: Optional[str] = None,
    video_url: str = "",
    output_dir: str = "",
    audio_mode: str = "local",
    # track_tqdm=False: show only our explicit stage progress below, not every
    # library's internal tqdm (huggingface_hub's cache-check bars would otherwise
    # surface as confusing "Downloading 0/0 B" bars that aren't real downloads).
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Gradio callback: run the full pipeline and return outputs.

    Returns a tuple (result_video, source_srt, translated_srt, status_text).
    Any intermediate failure is caught and reported via the status field so
    the UI does not just crash with a stack trace.
    """
    if eleven_api_token:
        return None, None, None, 'Configure ELEVEN_API_KEY in the operator .env before starting LangSwap; per-request tokens are not supported.'
    if not video_file and not (video_url or "").strip():
        return None, None, None, "No video file uploaded or YouTube URL entered."
    if video_file and (video_url or "").strip():
        return None, None, None, "Choose either the uploaded video or one YouTube URL."
    if audio_mode == "youtube-fr" and (not video_url or (target_language or "").lower() not in {"french", "fr"}):
        return None, None, None, "The existing YouTube dub needs a YouTube URL and French as target language."
    if audio_mode not in {"local", "youtube-fr"}:
        return None, None, None, "Select a supported dubbing mode."
    if audio_mode == "local" and tts_engine in {"openvoice", "kokoro"} and (target_language or "").lower() not in {"french", "fr"}:
        return None, None, None, "This local voice engine speaks French; select another voice engine for this language."

    try:
        _resolve_output_dir(output_dir)
        if video_url:
            progress(0.02, desc="Downloading the YouTube video")
            video_path, label = download_original(video_url, Path(BASE_DIR) / "youtube_downloads")
        else:
            video_path = Path(video_file).expanduser().resolve()
            label = video_path.stem
        if not video_path.is_file():
            return None, None, None, f"File not found: {video_path}"

        if audio_mode == "youtube-fr":
            progress(0.15, desc="Downloading YouTube's existing French dub")
            youtube_video, youtube_en, youtube_fr = download_existing_french(
                video_url, Path(BASE_DIR) / "youtube_downloads")
            exported = _export_results(youtube_video, youtube_en, youtube_fr,
                                       output_dir, label + "-youtube", target_language)
            return (*exported, f"YouTube French audio saved in {Path(exported[0]).parent}")

        public_id = _public_id(str(video_path))
        repo = LocalOnlyFileRepository(public_id, BASE_DIR)
        resolved_device = _resolve_device(device)

        progress(0.0, desc="Initializing pipeline")

        config = TranslationPipelineConfig(
            source_lang=source_language or None,
            target_lang=target_language,
            name=public_id,
            public_id=public_id,
            num_speakers=None,
            source_video_path=str(video_path),
            base_dir=BASE_DIR,
            device=resolved_device,
            tts_model=tts_engine,
            dubbing_algo=dubbing_algo,
            watermark=False,
            skip_diarization=skip_diarization,
            asr_backend=asr_backend,
            translation_backend=translation_backend,
        )

        pipeline = VideoTranslationPipeline(config=config, file_repository=repo)

        progress(0.15, desc="Speech-to-Text (ASR)")
        pipeline._generate_asr()

        progress(0.40, desc="Translating")
        pipeline._generate_translation()

        progress(0.60, desc="Text-to-Speech synthesis")
        pipeline._generate_speech()

        progress(0.80, desc="Mixing & dubbing")
        video_translation = pipeline._merge(pipeline.config.dubbing_algo)
        pipeline.video_translation = video_translation

        progress(0.95, desc="Writing subtitles")
        source_srt, translated_srt = pipeline.generate_srt_files()

        progress(1.0, desc="Done")

        result_video_path, exported_source, exported_target = _export_results(
            video_translation.processed_video.file_path, source_srt.file_path,
            translated_srt.file_path, output_dir, label, target_language)
        status = (
            f"Success on device={resolved_device}.\n"
            f"Output dir: {Path(result_video_path).parent}\n"
            f"Video: {result_video_path}"
        )
        return result_video_path, exported_source, exported_target, status

    except Exception as e:  # noqa: BLE001 — display to user, do not crash UI
        tb = traceback.format_exc()
        logger.exception("Pipeline failure")
        return None, None, None, f"Pipeline failed: {e}\n\n{tb}"


# ── Translation editor (Edit tab) ────────────────────────────────────────────
# Reopen a finished job from data/<id>/ and edit its segments — timing, speaker,
# recognised source text, translation — plus add / delete / reorder, then re-dub
# only the changed lines (langswap.editor). The segment list is built with
# @gr.render (a dynamic per-row UI) rather than a Dataframe: a Dataframe with
# dynamic rows does not reliably grow its rendered height when rows are
# added/removed (needs a relayout), and it cannot host per-row move/delete
# buttons. Field edits are stashed into a side State on blur (no re-render);
# structural ops (move/delete/add/apply) fold those edits in and re-render.


def _session_to_state(session: editor.EditorSession) -> list[dict]:
    """Snapshot the loaded segments as plain dicts (the editor's working state).
    ``key`` is the stable 1-based segment id used to reconcile edits; new rows
    added in the UI carry ``key=None`` so they are treated as inserts."""
    return [
        {"key": i + 1, "start": s.start, "end": s.end, "speaker": s.speaker,
         "source": s.text, "translation": s.translation}
        for i, s in enumerate(session.segments)
    ]


def _merge_edits(state: list[dict] | None, edits: dict | None) -> list[dict]:
    """Fold pending per-field edits (keyed by row index) into a fresh state copy."""
    merged = [dict(item) for item in (state or [])]
    for idx, fields in (edits or {}).items():
        idx = int(idx)
        if 0 <= idx < len(merged):
            merged[idx].update(fields)
    return merged


def _state_to_rows(state: list[dict]) -> list[list]:
    """Convert the working state to editor.apply_edits rows
    (``[#, start, end, speaker, source, translation]``; blank # = new segment)."""
    return [
        ["" if item.get("key") is None else item["key"],
         item["start"], item["end"], item["speaker"], item["source"], item["translation"]]
        for item in state
    ]


def _segment_choices(session: editor.EditorSession) -> list[tuple[str, int]]:
    return [
        (f"#{i + 1}  [{seg.start:.2f}–{seg.end:.2f}]  {seg.text[:40]}", i)
        for i, seg in enumerate(session.segments)
    ]


def _stash_field(idx: int, field: str):
    """Event handler: record one field edit into the side State (no re-render)."""
    def handler(value, edits):
        edits = dict(edits or {})
        row = dict(edits.get(idx, {}))
        row[field] = value
        edits[idx] = row
        return edits
    return handler


def _move_row(idx: int, direction: int):
    """Event handler: swap this segment's time slot with its neighbour and re-sort.
    Play order is time order here, so reordering means exchanging start/end; the
    two moved segments get re-dubbed on Apply."""
    def handler(state, edits):
        state = _merge_edits(state, edits)
        j = idx + direction
        if 0 <= idx < len(state) and 0 <= j < len(state):
            state[idx]["start"], state[j]["start"] = state[j]["start"], state[idx]["start"]
            state[idx]["end"], state[j]["end"] = state[j]["end"], state[idx]["end"]
            state.sort(key=lambda it: float(it["start"]))
        return state, {}
    return handler


def _delete_row(idx: int):
    """Event handler: drop this segment."""
    def handler(state, edits):
        state = _merge_edits(state, edits)
        if 0 <= idx < len(state):
            state.pop(idx)
        return state, {}
    return handler


def editor_refresh_jobs(base_dir: str):
    """List editable jobs under base_dir for the picker."""
    jobs = editor.list_jobs(base_dir or BASE_DIR)
    return gr.update(choices=jobs, value=(jobs[0] if jobs else None))


def editor_load_job(base_dir: str, public_id: str):
    """Open a job: populate the working state, segment picker and video."""
    if not public_id:
        return None, [], {}, gr.update(choices=[], value=None), None, "Pick a job to load."
    try:
        session = editor.load_job(base_dir or BASE_DIR, public_id)
    except Exception as e:  # noqa: BLE001 — surface to the UI
        logger.exception("Failed to load job")
        return None, [], {}, gr.update(choices=[], value=None), None, f"Failed to load {public_id}: {e}"

    state = _session_to_state(session)
    status = (
        f"Loaded job {public_id} — {len(state)} segment(s).\n"
        "Edit fields, ➕ add, 🗑 delete or ⬆/⬇ reorder, then 'Apply edits & re-dub'."
    )
    return (
        session,
        state,
        {},
        gr.update(choices=_segment_choices(session), value=None),
        session.result_video_path,
        status,
    )


def editor_preview_segment(session: "editor.EditorSession | None", index):
    """Return (original_audio, dubbed_audio) for the selected segment."""
    if session is None or index is None:
        return None, None
    seg = session.segments[int(index)]
    src = seg.source_file if seg.source_file and os.path.exists(seg.source_file) else None
    gen = seg.generated_file if seg.generated_file and os.path.exists(seg.generated_file) else None
    return src, gen


def editor_add_segment(state, edits):
    """Append a new (blank-key) segment after the last one."""
    state = _merge_edits(state, edits)
    last_end = max((float(item["end"]) for item in state), default=0.0)
    state.append({
        "key": None, "start": round(last_end, 2), "end": round(last_end + 2.0, 2),
        "speaker": "SPEAKER_00", "source": "", "translation": "",
    })
    return state, {}


def editor_apply_edits(session: "editor.EditorSession | None", state, edits):
    """Fold pending edits in, reconcile add/delete/reorder/edits and re-dub."""
    if session is None:
        return None, state, edits, gr.update(), None, None, None, "Load a job first."

    merged = _merge_edits(state, edits)

    def _outputs(new_state, status):
        return (
            session,
            new_state,
            {},
            gr.update(choices=_segment_choices(session), value=None),
            session.result_video_path,
            os.path.join(session.job_dir, "source_transcript.srt")
            if os.path.exists(os.path.join(session.job_dir, "source_transcript.srt")) else None,
            os.path.join(session.job_dir, "translated_transcript.srt")
            if os.path.exists(os.path.join(session.job_dir, "translated_transcript.srt")) else None,
            status,
        )

    try:
        session = editor.apply_edits(session, _state_to_rows(merged))
    except Exception as e:  # noqa: BLE001
        logger.exception("Re-dub failed")
        return _outputs(merged, f"Re-dub failed: {e}")

    return _outputs(
        _session_to_state(session),
        "Done — applied edits, re-dubbed changed/added segments and refreshed "
        "the video, subtitles and segment list.",
    )


def build_editor_tab() -> None:
    """Build the 'Edit translation' tab contents inside the current Blocks."""
    gr.Markdown(
        "## Edit a finished translation\n"
        "Reopen a job and edit any segment's **start/end time, speaker, recognised "
        "source text or translation**. Use **➕ Add segment** to insert a line, "
        "**🗑** to delete one, and **⬆ / ⬇** on a row to move it up/down (this swaps "
        "the two segments' time slots). Press *Apply* to re-dub **only the "
        "changed/added lines** — timing changes re-cut the voice reference from the "
        "original audio; the background music/effects bed is always reused."
    )
    session_state = gr.State(value=None)
    seg_state = gr.State(value=[])      # working list of segment dicts (drives the render)
    edits_state = gr.State(value={})    # pending field edits {row_index: {field: value}}

    with gr.Row():
        base_dir_in = gr.Textbox(value=BASE_DIR, label="Data directory", scale=1)
        job_dd = gr.Dropdown(
            choices=editor.list_jobs(BASE_DIR), label="Job (public_id)",
            scale=2, allow_custom_value=True,
        )
        refresh_btn = gr.Button("↻ Refresh", scale=0)
        load_btn = gr.Button("Load", variant="primary", scale=0)

    with gr.Row():
        with gr.Column(scale=3):
            @gr.render(inputs=[seg_state])
            def render_segments(state):
                if not state:
                    gr.Markdown("_No segments. Load a job, or click ➕ Add segment._")
                    return
                # Speakers already present in the job — offered as a dropdown, while
                # allow_custom_value still lets you assign a brand-new speaker label.
                speaker_choices = sorted({(it.get("speaker") or "SPEAKER_00") for it in state})
                for i, seg in enumerate(state):
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=0, min_width=34):
                                gr.Markdown(f"**{i + 1}**")
                            start_in = gr.Number(value=seg["start"], label="start", scale=1, min_width=90, interactive=True)
                            end_in = gr.Number(value=seg["end"], label="end", scale=1, min_width=90, interactive=True)
                            speaker_in = gr.Dropdown(
                                choices=speaker_choices, value=seg["speaker"], label="speaker",
                                allow_custom_value=True, scale=1, min_width=130, interactive=True,
                            )
                            up_btn = gr.Button("⬆", scale=0, min_width=42)
                            down_btn = gr.Button("⬇", scale=0, min_width=42)
                            del_btn = gr.Button("🗑", scale=0, min_width=42)
                        with gr.Row():
                            source_in = gr.Textbox(value=seg["source"], label="source (recognised)", scale=1, lines=2, interactive=True)
                            tr_in = gr.Textbox(value=seg["translation"], label="translation", scale=1, lines=2, interactive=True)

                    # Capture on .input (fires on every keystroke/selection, not just
                    # blur) so edits land in edits_state even if the user clicks Apply
                    # straight from a field — and it never re-renders (edits_state is
                    # not a render input).
                    start_in.input(_stash_field(i, "start"), inputs=[start_in, edits_state], outputs=[edits_state])
                    end_in.input(_stash_field(i, "end"), inputs=[end_in, edits_state], outputs=[edits_state])
                    speaker_in.input(_stash_field(i, "speaker"), inputs=[speaker_in, edits_state], outputs=[edits_state])
                    speaker_in.change(_stash_field(i, "speaker"), inputs=[speaker_in, edits_state], outputs=[edits_state])
                    source_in.input(_stash_field(i, "source"), inputs=[source_in, edits_state], outputs=[edits_state])
                    tr_in.input(_stash_field(i, "translation"), inputs=[tr_in, edits_state], outputs=[edits_state])
                    up_btn.click(_move_row(i, -1), inputs=[seg_state, edits_state], outputs=[seg_state, edits_state])
                    down_btn.click(_move_row(i, 1), inputs=[seg_state, edits_state], outputs=[seg_state, edits_state])
                    del_btn.click(_delete_row(i), inputs=[seg_state, edits_state], outputs=[seg_state, edits_state])

            with gr.Row():
                add_btn = gr.Button("➕ Add segment", scale=0)
                apply_btn = gr.Button("Apply edits & re-dub", variant="primary")
        with gr.Column(scale=2):
            result_video = gr.Video(label="Dubbed video")
            with gr.Accordion("Preview a segment (last applied state)", open=False):
                segment_dd = gr.Dropdown(choices=[], label="Segment", value=None)
                orig_audio = gr.Audio(label="Original (cloned source)", type="filepath")
                dubbed_audio = gr.Audio(label="Current dubbed segment", type="filepath")
            source_srt_dl = gr.File(label="Source transcript (.srt)")
            target_srt_dl = gr.File(label="Translated transcript (.srt)")
    editor_status = gr.Textbox(label="Status", lines=4)

    refresh_btn.click(editor_refresh_jobs, inputs=[base_dir_in], outputs=[job_dd])
    load_btn.click(
        editor_load_job,
        inputs=[base_dir_in, job_dd],
        outputs=[session_state, seg_state, edits_state, segment_dd, result_video, editor_status],
    )
    segment_dd.change(
        editor_preview_segment,
        inputs=[session_state, segment_dd],
        outputs=[orig_audio, dubbed_audio],
    )
    add_btn.click(editor_add_segment, inputs=[seg_state, edits_state], outputs=[seg_state, edits_state])
    apply_btn.click(
        editor_apply_edits,
        inputs=[session_state, seg_state, edits_state],
        outputs=[
            session_state, seg_state, edits_state, segment_dd, result_video,
            source_srt_dl, target_srt_dl, editor_status,
        ],
    )


def build_translate_tab() -> None:
    """Build the 'Translate' tab contents inside the current Blocks."""
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Vidéo locale", sources=["upload"])
            video_url = gr.Textbox(label="URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
            output_dir = gr.Textbox(label="Dossier de sortie", value=OUTPUT_DIR,
                                    info="Dossier accessible depuis la machine qui exécute LangSwap")
            no_request_token = gr.State("")
            audio_mode = gr.Radio([("Voix produite par LangSwap", "local"),
                                   ("Piste française de YouTube", "youtube-fr")],
                                  value="local", label="Version à produire")

            with gr.Row():
                source_lang = gr.Dropdown(
                    LANGUAGES, value="english", label="Langue source",
                    allow_custom_value=True,
                )
                target_lang = gr.Dropdown(
                    LANGUAGES, value="french", label="Langue de sortie",
                    allow_custom_value=True,
                )

            with gr.Accordion("Models / backends", open=False):
                tts_engine = gr.Dropdown(TTS_ENGINES, value=backends.default("tts"), label="TTS engine")
                asr_backend = gr.Dropdown(ASR_BACKENDS, value=backends.default("asr"), label="ASR backend")
                translation_backend = gr.Dropdown(
                    TRANSLATION_BACKENDS, value=backends.default("translation"), label="Translation backend"
                )
                dubbing_algo = gr.Dropdown(
                    DUBBING_ALGOS, value=backends.default("dubbing"), label="Dubbing algorithm"
                )
                device = gr.Dropdown(
                    ["auto", "cuda", "mps", "cpu"], value="auto", label="Device"
                )
                skip_diarization = gr.Checkbox(
                    value=False,
                    label="Skip speaker diarization (use only for a single speaker)",
                )

            run_btn = gr.Button("Traduire", variant="primary")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Vidéo traduite")
            source_srt_out = gr.File(label="Transcription source (.srt)")
            target_srt_out = gr.File(label="Sous-titres traduits (.srt)")
            status_out = gr.Textbox(label="Status", lines=10)

    run_btn.click(
        fn=translate_video,
        inputs=[
            video_in, source_lang, target_lang, tts_engine, asr_backend,
            translation_backend, dubbing_algo, device, skip_diarization,
            no_request_token, video_url, output_dir, audio_mode,
        ],
        outputs=[video_out, source_srt_out, target_srt_out, status_out],
    )

    gr.Markdown(
        "### Tips\n"
        "- First run auto-downloads the model weights into `models_weights/` — "
        "this can take a long time and needs free disk space.\n"
        "- Intermediate artifacts are cached under `data/<id>/` — re-running on "
        "the same file skips ASR/translation if their JSON outputs already exist."
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LangSwap — traduction vidéo") as demo:
        gr.Markdown(
            "# LangSwap — traduction vidéo\n"
            "Colle une URL YouTube ou importe une vidéo, puis choisis la langue et le dossier de sortie."
        )

        with gr.Tabs():
            with gr.Tab("Translate"):
                build_translate_tab()
            with gr.Tab("Edit translation"):
                build_editor_tab()

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the langswap Gradio demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the local data directory used for intermediate files.",
    )
    args = parser.parse_args()

    if args.data_dir:
        global BASE_DIR
        BASE_DIR = args.data_dir
        os.environ["LANGSWAP_DATA_DIR"] = args.data_dir

    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
