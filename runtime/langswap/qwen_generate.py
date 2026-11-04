"""Isolated Qwen3-TTS inference for the native single-narrator workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def generate(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(manifest["directory"])
    output.mkdir(parents=True, exist_ok=True)
    ref_audio = Path(manifest["reference_audio"])
    reference_hash = hashlib.sha256(ref_audio.read_bytes()).hexdigest()
    from qwen_tts import Qwen3TTSModel
    segments = manifest["segments"]
    prepared = []
    for index, segment in enumerate(segments):
        identity = {"model": MODEL, "engine_version": 1, "reference": reference_hash,
                    "reference_text": manifest["reference_text"], **segment}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
        audio = output / f"{index:04}.wav"
        sidecar = output / f"{index:04}.json"
        cached = False
        if audio.is_file() and sidecar.is_file():
            try:
                previous = json.loads(sidecar.read_text(encoding="utf-8"))
                cached_audio, cached_rate = sf.read(audio, dtype="float32")
                cached = (previous.get("identity_sha256") == digest
                          and previous.get("wav_sha256") == hashlib.sha256(audio.read_bytes()).hexdigest()
                          and cached_rate > 0 and cached_audio.size >= cached_rate // 10
                          and np.isfinite(cached_audio).all()
                          and np.max(np.abs(cached_audio)) > 1e-6)
            except (OSError, ValueError, RuntimeError):
                cached = False
        prepared.append((index, segment, digest, audio, sidecar, cached))
    if all(item[5] for item in prepared):
        print(f"LANGSWAP_STAGE|Voix Qwen : {len(prepared)}/{len(prepared)} phrases en cache", flush=True)
        return len(prepared)

    model = Qwen3TTSModel.from_pretrained(
        MODEL, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa")
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio), ref_text=manifest["reference_text"])
    for index, segment, digest, audio, sidecar, cached in prepared:
        if not cached:
            torch.manual_seed(1000 + index)
            waves, sample_rate = model.generate_voice_clone(
                text=segment["text"], language="French", voice_clone_prompt=prompt)
            wave = np.asarray(waves[0], dtype="float32").reshape(-1)
            if (wave.size < sample_rate // 10 or not np.isfinite(wave).all()
                    or np.max(np.abs(wave)) > 1.0 or np.max(np.abs(wave)) < 1e-6):
                raise ValueError(f"Qwen returned invalid audio at segment {index}.")
            temporary = output / f"{index:04}.wav.tmp"
            sf.write(temporary, wave, sample_rate, format="WAV")
            os.replace(temporary, audio)
            metadata = {"identity_sha256": digest,
                        "wav_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                        "sample_rate": sample_rate,
                        "duration": len(wave) / sample_rate, **segment}
            temp_json = output / f"{index:04}.json.tmp"
            temp_json.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_json, sidecar)
        if (index + 1) % 10 == 0 or index + 1 == len(segments):
            print(f"LANGSWAP_STAGE|Voix Qwen : {index+1}/{len(segments)} phrases", flush=True)
    return len(segments)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--manifest", type=Path, required=True)
    args = cli.parse_args()
    raise SystemExit(0 if generate(args.manifest) else 1)
