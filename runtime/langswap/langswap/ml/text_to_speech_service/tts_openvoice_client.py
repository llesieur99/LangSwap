"""Native French (Kokoro) + OpenVoice tone-color clone.

Kokoro speaks French. OpenVoice copies each speaker's timbre without
copying their English accent. One embedding per speaker label.
"""
from __future__ import annotations

import logging
import os

from langswap.ml.text_to_speech_service.speaker_voices import (
    build_speaker_references,
    file_digest,
    speaker_label,
    synthesize_cached_segment,
    validate_audio,
    validate_segment_intervals,
)
from langswap.ml.text_to_speech_service.tts_kokoro_client import KokoroClient
from langswap.model_config import MODEL_WEIGHTS_DIR

logger = logging.getLogger(__name__)

_CKPT_DIR = os.path.join(MODEL_WEIGHTS_DIR, "openvoice_v2")
_HF_REPO = "orionai/openvoice-v2"


def _ensure_checkpoints():
    conv = os.path.join(_CKPT_DIR, "converter", "checkpoint.pth")
    cfg = os.path.join(_CKPT_DIR, "converter", "config.json")
    if os.path.exists(conv) and os.path.exists(cfg) and os.path.getsize(conv) > 1_000_000:
        return _CKPT_DIR
    os.makedirs(os.path.join(_CKPT_DIR, "converter"), exist_ok=True)
    logger.info("Downloading OpenVoice v2 converter from HuggingFace %s", _HF_REPO)
    from huggingface_hub import hf_hub_download
    for name in ("converter/checkpoint.pth", "converter/config.json"):
        hf_hub_download(
            repo_id=_HF_REPO,
            filename=name,
            local_dir=_CKPT_DIR,
        )
    if not os.path.exists(conv):
        raise FileNotFoundError("OpenVoice converter checkpoint missing after HF download")
    return _CKPT_DIR


class OpenVoiceClient:
    def __init__(self, device: str = "cuda"):
        self.device = "cuda" if (device or "").startswith("cuda") else "cpu"
        self.sample_rate = 24000
        self._base = KokoroClient(device=self.device)
        self._conv = None
        self._refs = {}
        self._se = {}
        self._source_se = None

    def load_models(self):
        self._base.load_models()
        self.sample_rate = self._base.sample_rate
        if self._conv is not None:
            return
        ckpt = _ensure_checkpoints()
        from openvoice.api import ToneColorConverter

        cfg = os.path.join(ckpt, "converter", "config.json")
        pth = os.path.join(ckpt, "converter", "checkpoint.pth")
        logger.info("Loading OpenVoice converter %s", pth)
        self._conv = ToneColorConverter(cfg, device=self.device)
        self._conv.load_ckpt(pth)
        self._conv.watermark_model = None

    def prepare_speakers(self, video_translation, temp_folder):
        validate_segment_intervals(video_translation.translated_texts)
        self.load_models()
        vocals = (video_translation.background_audio or {}).get("vocals.wav")
        self._refs = build_speaker_references(
            vocals, video_translation.translated_texts,
            os.path.join(os.path.dirname(temp_folder), "speaker_refs"),
        )
        self._se = {}
        for speaker, path in self._refs.items():
            try:
                self._se[speaker] = self._extract_se(path)
            except Exception as exc:
                raise RuntimeError(f"OpenVoice reference extraction failed for {speaker!r}") from exc
        # A fixed base-voice calibration avoids a new source identity each segment.
        if self._source_se is None:
            calibration = os.path.join(os.path.dirname(temp_folder), "speaker_refs", "kokoro_base.wav")
            self._base.generate_audio(
                text="Bonjour, voici une phrase pour identifier la voix de narration.",
                source_audio_file="", source_text="", save_path=calibration, language="fr",
            )
            validate_audio(calibration)
            self._source_se = self._extract_se(calibration)

    def _extract_se(self, wav_path: str):
        return self._conv.extract_se(wav_path)

    def _convert(self, src_wav: str, tgt_se, dest: str):
        src_se = self._source_se
        if src_se is None:
            raise RuntimeError("OpenVoice base voice has not been calibrated")
        self._conv.convert(
            audio_src_path=src_wav,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=dest,
            message="@MyShell",
        )

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,
        save_path: str,
        language: str,
        duration=None,
        speaker: str = "SPEAKER_00",
    ):
        self.load_models()
        spk = speaker or "SPEAKER_00"
        target = self._se.get(spk)
        if target is None:
            raise RuntimeError(f"No OpenVoice reference prepared for speaker {spk!r}")
        tmp = save_path + ".kokoro.wav"
        try:
            self._base.generate_audio(
                text=text, source_audio_file=source_audio_file, source_text=source_text,
                save_path=tmp, language=language, duration=duration,
            )
            validate_audio(tmp)
            self._convert(tmp, target, save_path)
            validate_audio(save_path)
        except Exception as exc:
            raise RuntimeError(f"OpenVoice synthesis failed for speaker {spk!r}") from exc
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def tts_pipeline(self, video_translation, temp_folder, language="fr"):
        self.prepare_speakers(video_translation, temp_folder)
        from tqdm.auto import tqdm
        for segment in tqdm(video_translation.translated_texts, desc="Kokoro+OpenVoice", leave=True):
            speaker = speaker_label(segment)
            synthesize_cached_segment(
                segment, temp_folder, language, "openvoice",
                {"base": self._base.voice, "reference": file_digest(self._refs[speaker])},
                lambda path, segment=segment, speaker=speaker: self.generate_audio(
                    text=segment.translation, source_audio_file=self._refs[speaker],
                    source_text=segment.text, save_path=path, language=language,
                    duration=segment.end - segment.start, speaker=speaker,
                ),
            )
        return video_translation
