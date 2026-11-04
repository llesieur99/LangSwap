import torch
import torchaudio
import os
import wave
import hashlib
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from typing import Optional


class DemucsClient:

    def __init__(self, load_separator: bool = True):
        try:
            import demucs.api
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `demucs` required for audio separation. "
                "Install it with `pip install demucs`."
            ) from e
        # The separator model load takes several seconds.  merge_background()
        # never uses it, so callers on the merge-only path pass
        # load_separator=False; separate() lazily loads it if ever needed.
        self._separator = demucs.api.Separator() if load_separator else None

    def _ensure_separator(self):
        if self._separator is None:
            import demucs.api
            self._separator = demucs.api.Separator()

    def separate(self, audio_file_path: str, output_directory: str) -> list[tuple[str, str]]:
        background_files = []
        # check of already separated files — do not load Demucs if cache hits
        # (loading it would steal VRAM from OmniVoice on a 24 GB GPU).
        modes = ['bass', 'drums', 'other', 'vocals']
        try:
            demucs_version = version('demucs')
        except PackageNotFoundError:
            demucs_version = 'unversioned'
        digest = hashlib.sha256(f'demucs:{demucs_version}:Separator:default:pcm16\0'.encode('ascii'))
        with open(audio_file_path, 'rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        source_digest = digest.hexdigest()
        cache_manifest = Path(output_directory) / 'source.sha256'

        def valid_stem(mode):
            try:
                with wave.open(os.path.join(output_directory, f'{mode}.wav'), 'rb') as audio:
                    if audio.getnframes() == 0:
                        return False
                    audio.setpos(audio.getnframes() - 1)
                    return len(audio.readframes(1)) == audio.getnchannels() * audio.getsampwidth()
            except (OSError, EOFError, wave.Error):
                return False

        try:
            matches_source = cache_manifest.read_text(encoding='ascii').strip() == source_digest
        except (OSError, UnicodeError):
            matches_source = False
        if matches_source and all(valid_stem(mode) for mode in modes):
            for m in modes:
                file_name = f'{m}.wav'
                generated_file_path = os.path.join(output_directory, file_name)

                background_files.append((generated_file_path, file_name))
        else:
            os.makedirs(output_directory, exist_ok=True)
            cache_manifest.unlink(missing_ok=True)
            self._ensure_separator()
            separated = self._separator.separate_audio_file(audio_file_path)
            if not all(mode in separated[1] for mode in modes):
                raise RuntimeError('Demucs did not produce all four required stems')
            for file, source in separated[1].items():
                file_name = f'{file}.wav'
                generated_file_path = os.path.join(output_directory, file_name)
                import demucs.api
                demucs.api.save_audio(source, generated_file_path, samplerate=self._separator.samplerate)

                background_files.append((generated_file_path, file_name))
            if not all(valid_stem(mode) for mode in modes):
                raise RuntimeError('Demucs produced missing, empty or invalid WAV stems')
            cache_manifest.write_text(source_digest, encoding='ascii')
        return background_files

    def merge_background(self, generated_audio_path, audio_backgrounds: dict[str, str],
                        modes: Optional[list] = None,
                        normalize: bool = True,
                        target_peak: float = 0.95) -> tuple[torch.Tensor, int]:
        """
        Merge generated speech audio with background stems (drums, bass, other).

        Args:
            generated_audio_path: Path to the generated dubbed vocals
            audio_backgrounds: Dict mapping stem names to file paths
            modes: List of stems to include (default: drums, bass, other)
            normalize: Whether to apply peak normalization to prevent clipping
            target_peak: Target peak level for normalization (default 0.95 to leave headroom)

        Returns:
            Tuple of (merged audio tensor, sample rate)
        """
        if modes is None:
            modes = ['drums.wav', 'bass.wav', 'other.wav']

        speech_audio, speech_sr = torchaudio.load(generated_audio_path)

        if speech_audio.shape[0] > 1:
            speech_audio = torch.mean(speech_audio, 0, keepdim=True)

        audio = speech_audio
        for m in modes:
            sample_path = audio_backgrounds[m]

            back_sound, sr = torchaudio.load(sample_path)
            if back_sound.shape[0] > 1:
                back_sound = torch.mean(back_sound, 0, keepdim=True)
            if sr != speech_sr:
                back_sound = torchaudio.functional.resample(back_sound, sr, speech_sr)

            if back_sound.shape[1] > audio.shape[1]:
                back_sound = back_sound[:, :speech_audio.shape[1]]
            elif back_sound.shape[1] < audio.shape[1]:
                pause = back_sound.new_zeros((1, audio.shape[1] - back_sound.shape[1]))
                back_sound = torch.cat([back_sound, pause], dim=1)

            audio += back_sound

        # Apply peak normalization to prevent clipping
        if normalize:
            peak = torch.abs(audio).max()
            if peak > target_peak:
                audio = audio * (target_peak / peak)

        return audio, speech_sr
