import os

import torchaudio
from pydub import AudioSegment
from torchaudio import transforms
from tqdm import tqdm

from langswap.file_repository import FileRepository


class AudioDubbingManager:
    _file_repository: FileRepository

    def __init__(self, file_repository: FileRepository, device="cuda"):
        self._file_repository = file_repository

        self.device = device

    @classmethod
    def resample_save(self, audio_path: str, target_sr=16000):
        """
        Function resamples audio and rewrites the file.
        """
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0).unsqueeze(0)

        if sr != target_sr:
            transform = transforms.Resample(sr, target_sr)
            resampled_audio = transform(waveform)
            torchaudio.save(audio_path, resampled_audio, sample_rate=target_sr)
        return audio_path

    def split_audio_seconds(
        self, video_translation, audio_path, temp_folder, sample_rate=24000
    ):
        """
        Splits the audio in seconds mentioned in df.
        Audio fragments are then used for style transfering.

        Sample rate is set to the TTS engine setting (24000 for OmniVoice).
        """

        sound_file = AudioSegment.from_wav(audio_path)
        os.makedirs(temp_folder, exist_ok=True)

        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="Split audio pipeline.",
                leave=True,
            )
        ):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")
            # These cheap clips must follow the current vocal stem, including
            # after source replacement or an edit with unchanged timestamps.
            sound = sound_file[segment.start * 1000 : segment.end * 1000]
            sound = sound.set_frame_rate(sample_rate).set_channels(1)
            sound.export(file_path, format="wav")
            video_translation.translated_texts[idx].source_file = file_path
        return video_translation

    @classmethod
    def merge_audio_files(self, files: list, audio_path: str):
        merged_audio = AudioSegment.empty()

        for file in files:
            audio = AudioSegment.from_file(file)
            merged_audio += audio

        # Export the merged audio to a file
        merged_audio.export(audio_path, format="wav")
