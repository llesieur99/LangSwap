"""Qwen voice cache validates source, text, audio and sample rate."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

import qwen_generate


def test_regeneration_only_when_reference_or_text_changes(monkeypatch, tmp_path):
    calls = []
    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def create_voice_clone_prompt(self, **kwargs):
            return kwargs
        def generate_voice_clone(self, **kwargs):
            calls.append(kwargs["text"])
            return [np.full(2400, 0.2, dtype="float32")], 24000
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=FakeModel))
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.full(24000, 0.1, dtype="float32"), 24000)
    manifest = tmp_path / "manifest.json"
    data = {"reference_audio": str(reference), "reference_text": "Hello there.",
            "directory": str(tmp_path / "output"),
            "segments": [{"start": 1.0, "end": 3.0, "text": "Bonjour."}]}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    assert qwen_generate.generate(manifest) == 1
    assert qwen_generate.generate(manifest) == 1
    assert calls == ["Bonjour."]
    data["segments"][0]["text"] = "Bonsoir."
    manifest.write_text(json.dumps(data), encoding="utf-8")
    assert qwen_generate.generate(manifest) == 1
    assert calls == ["Bonjour.", "Bonsoir."]
    assert sf.info(tmp_path / "output" / "0000.wav").samplerate == 24000
