"""langswap command-line entrypoint.

Two subcommands:

    # Run the full pipeline on a LOCAL video file (no S3/AWS) — for dev/testing.
    python main.py local in.mp4 english russian
    python main.py local in.mp4 english --tts omnivoice --stop-after asr

    # Run the S3/RunPod-style pipeline from a JSON job file (needs AWS creds).
    python main.py runpod --input-file tests/fixtures/test_input.json

The actual pipeline logic lives in ``langswap.api`` (the importable library used
by ``serverless.py`` and the batch scripts); this file is only the CLI surface.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("langswap.main")


def _id_for(video_path: str) -> str:
    """Derive a stable ID from the input path so caches survive re-runs."""
    return hashlib.md5(str(video_path).encode("utf-8")).hexdigest()[:12]


def _pick_device(want: str) -> str:
    want = (want or "auto").lower()
    if want != "auto":
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_local(
    video: str,
    target_lang: str,
    source_lang: str | None,
    device: str,
    tts_engine: str,
    asr_backend: str,
    translation_backend: str,
    dubbing_algo: str,
    skip_diarization: bool,
    stop_after: str | None,
    num_speakers: int | None = None,
    speaker_voices: dict | None = None,
) -> int:
    """Run the pipeline against a local file, one stage at a time, with verbose
    logging.  Intermediate JSON outputs are cached under ``data/<id>/`` so reruns
    skip stages that already succeeded.  Returns a process exit code."""
    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        log.error("Video not found: %s", video_path)
        return 2

    from langswap.file_repository import LocalOnlyFileRepository
    from langswap.pipeline_models.models import TranslationPipelineConfig
    from langswap.translation_pipeline import VideoTranslationPipeline

    public_id = _id_for(str(video_path))
    base_dir = os.environ.get("LANGSWAP_DATA_DIR", "data")
    repo = LocalOnlyFileRepository(public_id, base_dir)

    resolved_device = _pick_device(device)
    log.info("video=%s id=%s device=%s data_dir=%s",
             video_path, public_id, resolved_device, repo.directory)

    config = TranslationPipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        name=public_id,
        public_id=public_id,
        num_speakers=num_speakers,
        source_video_path=str(video_path),
        base_dir=base_dir,
        device=resolved_device,
        tts_model=tts_engine,
        dubbing_algo=dubbing_algo,
        watermark=False,
        skip_diarization=skip_diarization,
        asr_backend=asr_backend,
        translation_backend=translation_backend,
        speaker_voices=speaker_voices or {},
    )

    pipeline = VideoTranslationPipeline(config=config, file_repository=repo)
    stages = [
        ("asr", pipeline._generate_asr),
        ("translation", pipeline._generate_translation),
        ("tts", pipeline._generate_speech),
        ("merge", lambda: setattr(pipeline, "video_translation",
                                  pipeline._merge(pipeline.config.dubbing_algo))),
        ("srt", pipeline.generate_srt_files),
    ]

    for name, fn in stages:
        log.info("===== stage start: %s =====", name)
        try:
            fn()
        except Exception:
            log.exception("Stage %s failed", name)
            return 1
        log.info("===== stage done:  %s =====", name)
        if stop_after == name:
            log.info("Stopping after %s as requested.", name)
            break

    out = pipeline.video_translation.processed_video
    if out and out.file_path:
        log.info("OUTPUT VIDEO: %s", out.file_path)
    log.info("OUTPUT DIR:   %s", repo.directory)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langswap", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # local --------------------------------------------------------------
    p_local = sub.add_parser("local", help="Run the pipeline on a local video file (no S3).")
    p_local.add_argument("video", help="Path to the source video.")
    p_local.add_argument("target_lang", nargs="?", default="french")
    p_local.add_argument("source_lang", nargs="?", default=None,
                         help="optional; pipeline auto-detects if omitted")
    p_local.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p_local.add_argument("--tts", default="openvoice")
    p_local.add_argument("--asr", default="vad")
    p_local.add_argument("--translation", default="marian")
    p_local.add_argument("--dubbing", default="speedup",
                         choices=["speedup", "stretch_whole", "pause_based"])
    diarization = p_local.add_mutually_exclusive_group()
    diarization.add_argument("--with-diarization", action="store_true", default=True,
                             help="identify narrators (enabled by default)")
    diarization.add_argument("--no-diarization", dest="with_diarization", action="store_false",
                             help="use one voice for a confirmed single-speaker video")
    p_local.add_argument("--speakers", type=int, default=None,
                         help="known positive speaker count; otherwise detect automatically")
    p_local.add_argument("--speaker-voice", action="append", default=[], metavar="LABEL=VOICE",
                         help="explicit stable voice per narrator for a preset TTS backend")
    p_local.add_argument("--stop-after", default=None,
                         choices=["asr", "translation", "tts", "merge", "srt"])

    # runpod -------------------------------------------------------------
    p_runpod = sub.add_parser("runpod", help="Run the S3/RunPod pipeline from a JSON job file.")
    p_runpod.add_argument("--input-file", default="tests/fixtures/test_input.json",
                          help="JSON file with a top-level 'input' key.")

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.command == "local":
        try:
            if args.speakers is not None and args.speakers < 1:
                raise ValueError("The speaker count must be positive.")
            voices = {}
            for item in args.speaker_voice:
                speaker, separator, voice = item.partition("=")
                if not separator or not speaker.strip() or not voice.strip():
                    raise ValueError("Use --speaker-voice LABEL=VOICE.")
                voices[speaker.strip()] = voice.strip()
            return run_local(
                video=args.video,
                target_lang=args.target_lang,
                source_lang=args.source_lang,
                device=args.device,
                tts_engine=args.tts,
                asr_backend=args.asr,
                translation_backend=args.translation,
                dubbing_algo=args.dubbing,
                skip_diarization=not args.with_diarization,
                stop_after=args.stop_after,
                num_speakers=args.speakers,
                speaker_voices=voices,
            )
        except Exception:
            log.exception("Fatal error")
            traceback.print_exc()
            return 1

    if args.command == "runpod":
        from langswap.api import test_video_translation_local
        test_video_translation_local(args.input_file)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
