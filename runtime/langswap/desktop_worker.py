"""Headless job entry point for the native Windows LangSwap application."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path

from gradio_demo import _export_results
from langswap.youtube_source import download_existing_track, download_original
from langswap.youtube_source import download_caption, caption_json3_to_srt, qwen_sentences_from_json3


def announce(stage: str) -> None:
    print(f"LANGSWAP_STAGE|{stage}", flush=True)


def run(args) -> tuple[str, str, str]:
    data_dir = Path(os.environ.get("LANGSWAP_DATA_DIR", "data")).resolve()
    output_dir = str(Path(args.output_dir).expanduser().resolve())
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.url:
        announce("Téléchargement de la vidéo")
        source, label = download_original(args.url, data_dir / "youtube_downloads")
    else:
        source = Path(args.file).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source introuvable : {source}")
        label = source.stem

    if args.mode == "youtube":
        if not args.url:
            raise ValueError("Une URL YouTube est requise pour la piste traduite existante.")
        announce("Piste audio et sous-titres YouTube")
        video, source_srt, translated_srt = download_existing_track(
            args.url, data_dir / "youtube_downloads", args.target_language)
        files = _export_results(video, source_srt, translated_srt,
                                output_dir, label + "-youtube", args.target_language)
        announce("Terminé")
        return tuple(Path(file).name for file in files)

    if args.mode == "qwen":
        if not args.url or args.target_language != "fr":
            raise ValueError("La voix Qwen demande une URL YouTube et le français en sortie.")
        return run_qwen(source, label, args, data_dir, output_dir)

    if args.target_language != "fr":
        raise ValueError("Le profil vocal local actuel produit du français.")
    if args.mode != "local":
        raise ValueError("Mode de doublage inconnu.")

    from langswap.file_repository import LocalOnlyFileRepository
    from langswap.pipeline_models.models import TranslationPipelineConfig
    from langswap.translation_pipeline import VideoTranslationPipeline

    public_id = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    repo = LocalOnlyFileRepository(public_id, str(data_dir / "jobs"))
    config = TranslationPipelineConfig(
        source_lang=args.source_language or None, target_lang="french",
        source_video_path=str(source), base_dir=str(data_dir / "jobs"),
        public_id=public_id, device="cuda", tts_model="openvoice",
        asr_backend="vad", translation_backend="marian", skip_diarization=False,
    )
    pipeline = VideoTranslationPipeline(config, repo)
    announce("Reconnaissance des paroles et des narrateurs")
    pipeline._generate_asr()
    announce("Traduction française")
    pipeline._generate_translation()
    announce("Création des voix")
    pipeline._generate_speech()
    announce("Montage de la vidéo")
    pipeline.video_translation = pipeline._merge(config.dubbing_algo)
    source_srt, translated_srt = pipeline.generate_srt_files()
    files = _export_results(
        pipeline.video_translation.processed_video.file_path,
        source_srt.file_path, translated_srt.file_path,
        output_dir, label, "french")
    announce("Terminé")
    return tuple(Path(file).name for file in files)


def run_qwen(source: Path, label: str, args, data_dir: Path, output_dir: str) -> tuple[str, str, str]:
    """Clone one narrator with Qwen, using existing French captions when present."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from scipy.signal import resample_poly

    from langswap.file_repository import LocalOnlyFileRepository
    from langswap.ml.text_to_speech_service.demucs_client import DemucsClient
    from langswap.pipeline_models.models import TranslationPipelineConfig
    from langswap.translation_pipeline import VideoTranslationPipeline

    public_id = hashlib.sha256((str(source) + ":qwen").encode("utf-8")).hexdigest()[:12]
    repo = LocalOnlyFileRepository(public_id, str(data_dir / "jobs"))
    input_video = source
    limit = getattr(args, "max_seconds", None)
    if limit:
        input_video = Path(repo.directory) / "preview_source.mp4"
        if not input_video.is_file():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(source), "-t", str(limit), "-c", "copy", str(input_video)], check=True)
    config = TranslationPipelineConfig(
        source_lang=args.source_language or None, target_lang="french",
        source_video_path=str(input_video), base_dir=str(data_dir / "jobs"),
        public_id=public_id, device="cuda", tts_model="openvoice",
        asr_backend="vad", translation_backend="marian", skip_diarization=True,
        num_speakers=1,
    )
    pipeline = VideoTranslationPipeline(config, repo)
    announce("Préparation de la voix originale")
    pipeline._generate_asr()
    vt = pipeline.video_translation
    vocals = Path(vt.background_audio["vocals.wav"])
    candidates = [s for s in vt.recognized_texts if s.text.strip() and 3 <= s.end-s.start <= 14]
    if not candidates:
        raise ValueError("Aucune parole anglaise assez longue pour cloner la voix.")
    reference = sorted(candidates, key=lambda s: s.start)[0]
    ref_audio = Path(repo.directory) / "qwen_reference.wav"
    if not ref_audio.is_file():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", str(reference.start), "-i", str(vocals),
                        "-t", str(reference.end-reference.start), "-ac", "1", "-ar", "24000",
                        str(ref_audio)], check=True)

    captions_dir = data_dir / "youtube_downloads"
    english = Path(repo.directory) / "qwen-source.srt"
    try:
        caption_json3_to_srt(download_caption(args.url, captions_dir, "en-orig"), english)
    except RuntimeError:
        pipeline._generate_translation()
        english, _ = pipeline.generate_srt_files()
        english = Path(english.file_path)
    french = Path(repo.directory) / "qwen-french.srt"
    try:
        french_caption = download_caption(args.url, captions_dir, "fr-orig")
        sentences = qwen_sentences_from_json3(french_caption, french)
    except RuntimeError:
        announce("Traduction locale des paroles")
        if not vt.translated_texts:
            pipeline._generate_translation()
        sentences = [dict(start=s.start, end=s.end, text=s.translation)
                     for s in pipeline.video_translation.translated_texts]
        _, translated = pipeline.generate_srt_files()
        french = Path(translated.file_path)

    if limit:
        sentences = [s for s in sentences if s["start"] < limit]
        for sentence in sentences:
            sentence["end"] = min(sentence["end"], limit)
        if not sentences:
            raise ValueError("Aucune phrase française dans cet extrait.")
    manifest = Path(repo.directory) / "qwen_manifest.json"
    manifest.write_text(json.dumps({"reference_audio": str(ref_audio),
                                    "reference_text": reference.text,
                                    "directory": str(Path(repo.directory) / "qwen_audio"),
                                    "segments": sentences}, ensure_ascii=False), encoding="utf-8")
    from langswap.model_pool import clear_pool
    clear_pool()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    announce("Clonage de la voix française")
    process = subprocess.Popen(["/opt/qwen-venv/bin/python", "-u", "/app/qwen_generate.py",
                                "--manifest", str(manifest)], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace")
    last_lines = []
    for line in process.stdout:
        if line.startswith("LANGSWAP_STAGE|"):
            print(line.rstrip(), flush=True)
        elif line.strip():
            last_lines.append(line.strip())
            last_lines = last_lines[-4:]
    if process.wait() != 0:
        raise RuntimeError("Le modèle Qwen a échoué. " + " ".join(last_lines[-2:]))

    announce("Synchronisation des voix et de la vidéo")
    source_audio, source_sr = sf.info(vocals).frames, sf.info(vocals).samplerate
    dubbed = np.zeros(source_audio, dtype="float32")
    review = []
    for index, sentence in enumerate(sentences):
        generated = Path(repo.directory) / "qwen_audio" / f"{index:04}.wav"
        wave, rate = sf.read(generated, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave = resample_poly(wave, source_sr, rate).astype("float32")
        first = round(sentence["start"] * source_sr)
        last = min(round(sentence["end"] * source_sr), source_audio)
        slot = last-first
        if slot <= 0:
            continue
        if len(wave) > slot:
            factor = len(wave)/slot
            from pyrubberband.pyrb import time_stretch
            wave = np.asarray(time_stretch(wave, source_sr, rate=factor), dtype="float32")
            if factor > 1.6:
                review.append({"segment": index, "speed_factor": round(factor, 2)})
        dubbed[first:first+min(slot,len(wave))] += wave[:slot]
    if not np.isfinite(dubbed).all() or not np.any(dubbed):
        raise ValueError("La piste de doublage produite est vide ou invalide.")
    styled = Path(repo.directory) / "qwen_voice.wav"
    sf.write(styled, dubbed, source_sr)
    merged, final_rate = DemucsClient(load_separator=False).merge_background(
        str(styled), vt.background_audio)
    final_audio = Path(repo.directory) / "qwen_mixed.wav"
    torchaudio.save(str(final_audio), merged, final_rate)
    result = Path(repo.directory) / "qwen_result.mp4"
    from langswap.ml.ffmpeg import FFmpegClient
    FFmpegClient().replace_audio(str(input_video), str(final_audio), str(result))
    (Path(repo.directory)/"qwen_review.json").write_text(json.dumps(review), encoding="utf-8")
    exports = _export_results(result, english, french, output_dir,
                              label + "-qwen" + ("-preview" if limit else ""), "french")
    announce("Terminé")
    return tuple(Path(file).name for file in exports)


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--file")
    parser.add_argument("--mode", choices=("local", "youtube", "qwen"), default="local")
    parser.add_argument("--source-language", choices=("english", ""), default="english")
    parser.add_argument("--target-language", choices=("fr", "es", "de", "it", "pt"), default="fr")
    parser.add_argument("--output-dir", default=os.environ.get("LANGSWAP_OUTPUT_DIR", "output"))
    parser.add_argument("--max-seconds", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        result = run(args)
        print("LANGSWAP_RESULT|" + "|".join(result), flush=True)
        return 0
    except Exception as exc:
        print(f"LANGSWAP_ERROR|{exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
