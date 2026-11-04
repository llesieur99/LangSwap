import torchaudio
import torch
from langswap.file_repository import FileRepository
from pyrubberband.pyrb import time_stretch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


# Silero VAD is a small, read-only model reused across every merge (and again on
# the re-dub / update-translation path).  Load it once per process instead of on
# each VideoDubbingManager construction.
_VAD_MODEL = None


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


# Keep original vocals inside long untranscribed gaps: Silero VAD can miss
# speech bridged by music, and such a gap's voice would otherwise be dropped
# entirely (vocals removed by Demucs, no TTS to replace them).  Fill the gap
# with the source vocals only when the gap is long enough and actually
# voice-like (energy check), otherwise keep silence.
_GAP_KEEP_MIN_SECONDS = 2.0
_GAP_KEEP_MIN_ENERGY_RATIO = 0.5  # gap RMS vs global vocals RMS


class VideoDubbingManager:
    _file_repository: FileRepository

    def __init__(self, file_repository: FileRepository, logger):
        self._file_repository = file_repository

        self.logger = logger

        self.model_vad = _get_vad_model()

    def _fill_gap_with_vocals(self, pause, vocals, vocals_sr, out_sr,
                              gap_start, gap_length, vocals_energy):
        """Fill an inter-segment silence gap with the original vocals when it
        looks like speech VAD missed (long gap + voice-like energy), placed at
        the end of the pause so it lines up with the next segment."""
        if gap_length < _GAP_KEEP_MIN_SECONDS:
            return pause
        i0 = int(gap_start * vocals_sr)
        i1 = min(int((gap_start + gap_length) * vocals_sr), vocals.shape[1])
        if i1 <= i0:
            return pause
        window = vocals[0, i0:i1]
        energy = window.float().pow(2).mean().sqrt()
        if energy < _GAP_KEEP_MIN_ENERGY_RATIO * vocals_energy:
            return pause
        if out_sr != vocals_sr:
            window = torchaudio.functional.resample(
                window.unsqueeze(0), vocals_sr, out_sr).squeeze(0)
        keep = min(window.shape[0], pause.shape[1])
        pause[0, pause.shape[1] - keep:] = window[:keep]
        return pause

    def get_pause(self, wav_path, sr, seconds=True):
        """
        Calculates the intra pauses in the audio
        """

        # hardcoded - change before that
        if "enhanced_audio" in wav_path:
            wav_path = wav_path.replace("enhanced_audio", "splitted_audio")
            wav_path = wav_path.replace("_enhanced", "")

        wav = read_audio(wav_path)
        speech_timesteps = get_speech_timestamps(wav, self.model_vad, return_seconds=seconds)
        # print(f"{wav_path}, {len(speech_timesteps)} speech timesteps are {speech_timesteps}")
        pause_long = 0
        # where does it happen that we do not have speech timesteps?
        if speech_timesteps:
            speech_timesteps[0]['pause'] = 0
            for t_idx in range(1, len(speech_timesteps)):
                speech_timesteps[t_idx]['pause'] = speech_timesteps[t_idx]['start'] - speech_timesteps[t_idx - 1]['end']

            pause_long = sum([i['pause'] for i in speech_timesteps])

        return pause_long, speech_timesteps

    def merge_pauses(self, wav_path, sr, timesteps):
        """
        Merges the audio back with pauses according to the timesteps
        """
        wav = read_audio(wav_path, sampling_rate=sr)

        # audio_samples = [wav[timesteps[0]['start']: timesteps[0]['end'] + 1].unsqueeze(0)]
        start = int(timesteps[0]['start'] * sr)
        end = int((timesteps[0]['end']) * sr)
        audio_samples = [wav[start: end].unsqueeze(0)]

        for audio_info in timesteps[1:]:
            pause = torch.zeros((1, round(audio_info['pause'] * sr)))
            audio_samples.append(pause)

            audio = wav[int(audio_info['start'] * sr): int((audio_info['end']) * sr)].unsqueeze(0)
            audio_samples.append(audio)

        audio_final = torch.cat(audio_samples, dim=1)
        self.logger.file_logger.debug(f"Changed audio length is from {wav.shape[0] / sr} to {audio_final.shape[1] / sr} for file {wav_path}")

        return audio_final

    def change_pauses(self, wav_gen_path: str, wav_source_path: str,
                    sr_gen: int, sr_source: int,
                    fallback_reduction: float = 0.8,
                    max_expansion: float = 1.5):
        """
        Adjusts intra-utterance pauses in generated audio to match source audio timing.

        Algorithm:
        1. Detect pauses in both source and generated audio using VAD
        2. Compute pause adjustment ratio ρ = (τ_gen - τ_src) / τ_gen
        3. If ρ > 0: compress pauses (generated has more pauses than source)
           If ρ < 0: expand pauses (generated has fewer pauses than source)
           If ρ = 1: use fallback (source has no pauses)
        4. Apply adjustment to each detected pause

        Args:
            wav_gen_path: Path to generated audio file
            wav_source_path: Path to source audio file
            sr_gen: Sample rate of generated audio
            sr_source: Sample rate of source audio
            fallback_reduction: Reduction factor when source has no pauses (default 0.8)
                               Empirically chosen to preserve some naturalness while reducing duration
            max_expansion: Maximum pause expansion factor to prevent unnatural gaps (default 1.5)

        Returns:
            Adjusted audio tensor
        """
        pause_dur_source, timesteps_source = self.get_pause(wav_source_path, sr_source, seconds=True)
        pause_dur_gen, timesteps_gen = self.get_pause(wav_gen_path, sr_gen, seconds=True)

        if pause_dur_gen:
            # Compute pause adjustment ratio
            # ρ > 0: need to compress pauses (generated longer)
            # ρ < 0: need to expand pauses (generated shorter)
            # ρ = 1: source has no pauses, use fallback
            pause_reduction = (pause_dur_gen - pause_dur_source) / pause_dur_gen

            if pause_reduction == 1.0:
                # Source has no pauses - use fallback to avoid removing all pauses
                # 0.8 preserves 20% of original pauses for naturalness
                pause_reduction = fallback_reduction
                self.logger.file_logger.debug(f"Source has no pauses, using fallback reduction {fallback_reduction}")

            # Apply adjustment to each pause
            for sp_idx, sp_data in enumerate(timesteps_gen):
                new_pause = sp_data['pause'] * (1 - pause_reduction)

                # Limit expansion to prevent unnatural long gaps
                if pause_reduction < 0:  # expansion case
                    new_pause = min(new_pause, sp_data['pause'] * max_expansion)

                sp_data['pause'] = max(new_pause, 0)  # ensure non-negative

            audio = self.merge_pauses(wav_gen_path, sr_gen, timesteps_gen)
            self.logger.file_logger.debug(
                f"Pause adjustment: src={pause_dur_source:.3f}s, gen={pause_dur_gen:.3f}s, "
                f"ρ={pause_reduction:.3f}, action={'compress' if pause_reduction > 0 else 'expand'}"
            )
        else:
            audio, sr_gen = torchaudio.load(wav_gen_path)
            self.logger.file_logger.debug(f"No pauses detected in generated audio, skipping adjustment")

        return audio

    def fit_segment(self, segment, source_sr: int, available_pause: float = 0.0):
        """Fit one complete phrase to its window, borrowing half the following gap."""
        import math
        start, end = float(segment.start), float(segment.end)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid dubbed segment timestamps.")
        available_pause = float(available_pause)
        if not math.isfinite(available_pause):
            raise ValueError("The available pause must be finite.")
        available_pause = max(0.0, available_pause)
        _, sr = torchaudio.load(segment.generated_file)
        audio = self.change_pauses(segment.generated_file, segment.source_file,
                                   sr_gen=sr, sr_source=source_sr).float().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio.mean(dim=0, keepdim=True)
        if not audio.numel() or not torch.isfinite(audio).all():
            raise ValueError("Invalid generated speech audio.")
        duration = audio.shape[1] / sr
        target_duration = end - start + min(max(duration - (end - start), 0.0), available_pause / 2)
        if duration > target_duration:
            rate = duration / target_duration
            fitted = time_stretch(audio[0].numpy(), sr, rate=rate)
            audio = torch.as_tensor(fitted, dtype=torch.float32).reshape(1, -1)
            target_frames = round(target_duration * sr)
            # Rubberband rounds output lengths; keep the clock within one sample.
            if audio.shape[1] < target_frames:
                audio = torch.nn.functional.pad(audio, (0, target_frames - audio.shape[1]))
            else:
                audio = audio[:, :target_frames]
        pause_after = max((end - start) + available_pause - audio.shape[1] / sr, 0.0)
        return audio, sr, pause_after

    def _merge_on_source_timeline(self, video_translation, vocals_audio, borrow_pause=True):
        """Place phrases on source timestamps without stretching the whole track."""
        import math
        source, source_sr = torchaudio.load(vocals_audio)
        source = source.float().cpu().mean(dim=0, keepdim=True)
        if not source.numel() or not torch.isfinite(source).all():
            raise ValueError("Invalid source vocals audio.")
        duration = source.shape[1] / source_sr
        output = torch.zeros_like(source)
        segments = sorted(video_translation.translated_texts, key=lambda s: s.start)
        if not segments:
            raise ValueError("No translated speech segments are available to dub.")
        rows = []
        covered_until = 0.0
        energy = source.pow(2).mean().sqrt()

        def restore_gap(start, end):
            if end <= start:
                return
            i0, i1 = round(start * source_sr), min(round(end * source_sr), source.shape[1])
            pause = output[:, i0:i1].clone()
            filled = self._fill_gap_with_vocals(pause, source, source_sr, source_sr,
                                              start, end - start, energy)
            output[:, i0:i1] = filled
            if torch.any(filled):
                self.logger.file_logger.warning(
                    f"Original source vocals retained in untranslated gap {start:.3f}-{end:.3f}s; review the transcript.")

        for idx, segment in enumerate(segments):
            start, end = float(segment.start), float(segment.end)
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.05):
                raise ValueError("A dubbed segment lies outside the source audio timeline.")
            next_start = float(segments[idx + 1].start) if idx + 1 < len(segments) else duration
            gap = max(next_start - end, 0.0) if borrow_pause else 0.0
            audio, rate, _ = self.fit_segment(segment, source_sr, gap)
            if rate != source_sr:
                audio = torchaudio.functional.resample(audio, rate, source_sr)
            i0 = round(start * source_sr)
            allowed = min(source.shape[1] - i0, round((end - start + gap / 2) * source_sr))
            if audio.shape[1] > allowed:
                if allowed <= 0:
                    raise ValueError("No source time remains for a dubbed phrase.")
                fitted = time_stretch(audio[0].numpy(), source_sr, rate=audio.shape[1] / allowed)
                audio = torch.as_tensor(fitted, dtype=torch.float32).reshape(1, -1)
            n = min(audio.shape[1], allowed)
            if start > covered_until:
                restore_gap(covered_until, start)
            output[:, i0:i0 + n] += audio[:, :n]
            generated_end = (i0 + n) / source_sr
            covered_until = max(covered_until, generated_end, end)
            rows.append({"start": start, "end": end, "generated_end": generated_end,
                         "generated_audio_length": n / source_sr})
        restore_gap(min(covered_until, duration), duration)
        peak = float(output.abs().max())
        if peak > 0.99:
            output *= 0.99 / peak
        self.logger.log_json(data=rows, file_name="pauses_for_stretch_whole.json")
        return output, source_sr

    def merge_timestamps_speedup(self, video_translation, vocals_audio):
        return self._merge_on_source_timeline(video_translation, vocals_audio, borrow_pause=True)

    def merge_timestamps_pause_based(self, video_translation, vocals_audio):
        return self._merge_on_source_timeline(video_translation, vocals_audio, borrow_pause=False)

    def merge_timestamps_stretch_whole(self, video_translation, vocals_audio):
        # Legacy selector retained for old jobs; stretching the full track caused drift.
        return self._merge_on_source_timeline(video_translation, vocals_audio, borrow_pause=False)
