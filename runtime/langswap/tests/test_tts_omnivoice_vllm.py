import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


def _install_fake_vllm_omni(monkeypatch):
    captured = {}

    class FakeSamplingParams:
        pass

    class FakeOmni:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def generate(self, prompt, sampling_params_list=None, use_tqdm=None):
            captured["prompt"] = prompt
            captured["sampling_params_list"] = sampling_params_list
            captured["use_tqdm"] = use_tqdm
            return [
                SimpleNamespace(
                    request_output=SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                multimodal_output={
                                    "audio": np.array([[0.0, 0.1, -0.1, 0.0]], dtype=np.float32),
                                    "sr": 22050,
                                }
                            )
                        ]
                    )
                )
            ]

        def close(self):
            captured["closed"] = True

    vllm_omni_module = types.ModuleType("vllm_omni")
    entrypoints_module = types.ModuleType("vllm_omni.entrypoints")
    omni_module = types.ModuleType("vllm_omni.entrypoints.omni")
    omni_module.Omni = FakeOmni
    inputs_module = types.ModuleType("vllm_omni.inputs")
    data_module = types.ModuleType("vllm_omni.inputs.data")
    data_module.OmniDiffusionSamplingParams = FakeSamplingParams

    monkeypatch.setitem(sys.modules, "vllm_omni", vllm_omni_module)
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints", entrypoints_module)
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.omni", omni_module)
    monkeypatch.setitem(sys.modules, "vllm_omni.inputs", inputs_module)
    monkeypatch.setitem(sys.modules, "vllm_omni.inputs.data", data_module)

    return captured, FakeSamplingParams


def _load_omnivoice_module(monkeypatch):
    tqdm_module = types.ModuleType("tqdm")
    tqdm_auto_module = types.ModuleType("tqdm.auto")
    tqdm_auto_module.tqdm = lambda iterable=None, **kwargs: iterable
    langswap_module = types.ModuleType("langswap")
    utils_module = types.ModuleType("langswap.utils")
    ml_processing_module = types.ModuleType("langswap.utils.ml_processing")
    mapper_module = types.ModuleType("langswap.utils.ml_processing.lang2code_mapper")
    mapper_module.map_language_to_code = lambda language, system="whisper": {"russian": "ru"}.get(language, language)

    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)
    monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto_module)
    monkeypatch.setitem(sys.modules, "langswap", langswap_module)
    monkeypatch.setitem(sys.modules, "langswap.utils", utils_module)
    monkeypatch.setitem(sys.modules, "langswap.utils.ml_processing", ml_processing_module)
    monkeypatch.setitem(sys.modules, "langswap.utils.ml_processing.lang2code_mapper", mapper_module)

    module_path = Path(__file__).resolve().parents[1] / "langswap" / "ml" / "text_to_speech_service" / "tts_omnivoice_client.py"
    spec = importlib.util.spec_from_file_location("test_tts_omnivoice_client", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_omnivoice_vllm_generate_audio_builds_prompt(monkeypatch, tmp_path):
    captured, fake_sampling_params_cls = _install_fake_vllm_omni(monkeypatch)

    module = _load_omnivoice_module(monkeypatch)

    reference_audio_path = tmp_path / "reference.wav"
    reference_audio = np.stack(
        [
            np.array([0.0, 0.1, 0.0, -0.1], dtype=np.float32),
            np.array([0.0, -0.1, 0.0, 0.1], dtype=np.float32),
        ],
        axis=1,
    )
    sf.write(reference_audio_path, reference_audio, 16000)

    output_path = tmp_path / "generated.wav"
    client = module.OmniVoiceClient()
    client.generate_audio(
        text="Privet mir",
        source_audio_file=str(reference_audio_path),
        source_text="Hello world",
        save_path=str(output_path),
        language="russian",
        duration=1.5,
    )
    client.__exit__(None, None, None)

    assert captured["init_kwargs"]["trust_remote_code"] is True
    assert Path(captured["init_kwargs"]["stage_configs_path"]).exists()

    prompt = captured["prompt"]
    assert prompt["prompt"] == "Privet mir"
    assert prompt["mm_processor_kwargs"]["ref_text"] == "Hello world"
    assert prompt["mm_processor_kwargs"]["sample_rate"] == 16000
    assert prompt["mm_processor_kwargs"]["duration"] == 1.5
    assert prompt["mm_processor_kwargs"]["lang"] == "ru"

    reference_audio_payload, payload_sr = prompt["multi_modal_data"]["audio"]
    assert payload_sr == 16000
    assert reference_audio_payload.dtype == np.float32
    assert reference_audio_payload.ndim == 1

    assert len(captured["sampling_params_list"]) == 1
    assert isinstance(captured["sampling_params_list"][0], fake_sampling_params_cls)
    assert captured["use_tqdm"] is False
    assert output_path.exists()
    rendered_audio, rendered_sr = sf.read(output_path, dtype="float32")
    assert rendered_sr == 22050
    assert rendered_audio.shape[0] == 4
    assert captured["closed"] is True


def test_omnivoice_vllm_auto_voice_omits_reference_audio(monkeypatch, tmp_path):
    captured, _ = _install_fake_vllm_omni(monkeypatch)

    module = _load_omnivoice_module(monkeypatch)

    output_path = tmp_path / "auto.wav"
    client = module.OmniVoiceClient()
    client.generate_audio(
        text="Hello from auto voice",
        source_audio_file="",
        source_text="",
        save_path=str(output_path),
        language="en",
    )

    prompt = captured["prompt"]
    assert prompt["prompt"] == "Hello from auto voice"
    assert "multi_modal_data" not in prompt
    assert prompt["mm_processor_kwargs"]["lang"] == "en"
    assert output_path.exists()
