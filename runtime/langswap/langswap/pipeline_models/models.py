from __future__ import annotations

import attr
import json

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Union, Literal, Optional


@attr.s(auto_attribs=True)
class RemoteFile:
    name: Optional[str] = attr.ib(default=None)
    file_path: Optional[str] = attr.ib(default=None)
    s3_url: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True)
class TextedSegment:
    text: str
    start: float
    end: float
    speaker: str


@attr.s(auto_attribs=True)
class TranslatedTextedSegment:
    text: str
    start: float
    end: float
    translation: str
    source_file: str
    generated_file: str
    speaker: str


@attr.s(auto_attribs=True)
class VideoTranslation:
    public_id: str
    source_lang_code: Optional[str] = attr.ib(default=None)
    source_file: Optional[RemoteFile] = attr.ib(default=None)
    extracted_audio: Optional[RemoteFile] = attr.ib(default=None)
    vad_filtered_audio: Optional[RemoteFile] = attr.ib(default=None)
    background_audio: dict[str, RemoteFile] = attr.field(factory=dict)
    recognized_texts: list[TextedSegment] = attr.field(factory=list)
    translated_texts: list[TranslatedTextedSegment] = attr.field(factory=list)
    processed_video: Optional[RemoteFile] = attr.ib(default=None)


@dataclass
class TranslationPipelineConfig:
    source_lang: str
    target_lang: str
    source_video_path: Union[Path, str]
    base_dir: Union[Path, str]
    public_id: str
    num_speakers: int = field(default=None)
    device: str = field(default="cuda")
    name: str = field(default="example")
    dubbing_algo: Literal["speedup", "pause_based", "stretch_whole"] = field(default="speedup")
    tts_model: Literal["openvoice", "kokoro", "chatterbox", "xtts", "omnivoice", "edge", "elevenlabs"] = field(default="openvoice")
    watermark: bool = field(default=False)
    skip_diarization: bool = field(default=False)
    asr_backend: str = field(default="vad")  # "vad" | "openai"
    translation_backend: str = field(default="marian")  # "marian" | "llamacpp" | "openai"
    speaker_voices: dict[str, str] = field(default_factory=dict)




def save_config_to_json(config: TranslationPipelineConfig, file_path: Union[Path, str]):

    config_dict = asdict(config)

    for field in ['source_video_path', 'base_dir']:
        if field in config_dict and isinstance(config_dict[field], Path):
            config_dict[field] = str(config_dict[field])

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4)


def load_config_from_json(file_path: Union[Path, str]) -> TranslationPipelineConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    for field in ['source_video_path', 'base_dir']:
        if field in config_dict:
            config_dict[field] = Path(config_dict[field])

    return TranslationPipelineConfig(**config_dict)


@dataclass
class TraslationUpdate:
    index: int
    text: str

    @classmethod
    def from_pairs(cls, pairs: List[Tuple[int, str]]) -> List["TraslationUpdate"]:
        return [cls(index=index, text=text) for index, text in pairs]
