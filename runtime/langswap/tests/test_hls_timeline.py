"""HLS chunks must occupy consecutive decode timelines for every track."""
import math
import shutil
import struct
import subprocess

import pytest

from langswap.hls import split_fragmented_mp4
from langswap.ml.ffmpeg import FFmpegClient


def box(kind, payload, extended=False):
    if extended:
        return struct.pack('>I4sQ', 1, kind.encode(), len(payload) + 16) + payload
    return struct.pack('>I4s', len(payload) + 8, kind.encode()) + payload


def init_segment(version=0):
    tracks = []
    for track_id, rate in [(1, 100000), (2, 48000)]:
        dates = bytes(16 if version == 1 else 8)
        tkhd = bytes([version, 0, 0, 0]) + dates + struct.pack('>I', track_id) + bytes(36)
        mdhd = bytes([version, 0, 0, 0]) + dates + struct.pack('>I', rate) + bytes(12)
        tracks.append(box('trak', box('tkhd', tkhd) + box('mdia', box('mdhd', mdhd))))
    return box('ftyp', b'isom') + box('moov', b''.join(tracks))


def media_segment(version=1, base=5, track_id=1, extended=False):
    trafs = []
    for track in [track_id, 2]:
        tfhd = box('tfhd', bytes(4) + struct.pack('>I', track))
        tfdt = box('tfdt', bytes([version, 0, 0, 0]) + struct.pack('>Q' if version else '>I', base), extended=extended)
        trafs.append(box('traf', tfhd + tfdt))
    return box('moof', b''.join(trafs)) + box('mdat', b'encoded samples remain unchanged')


def decode_times(data):
    """Independent recursive decoder for the time fields under test."""
    values = []

    def visit(start, end):
        while start < end:
            size, kind = struct.unpack_from('>I4s', data, start)
            header = 8
            if size == 1:
                size = struct.unpack_from('>Q', data, start + 8)[0]
                header = 16
            payload = start + header
            if kind in (b'moof', b'traf'):
                visit(payload, start + size)
            elif kind == b'tfdt':
                values.append(struct.unpack_from('>Q' if data[payload] else '>I', data, payload + 4)[0])
            start += size

    visit(0, len(data))
    return values


@pytest.mark.parametrize('version', [0, 1])
@pytest.mark.parametrize('extended', [False, True])
def test_offsets_use_each_track_timescale_without_changing_samples(tmp_path, version, extended):
    initial = init_segment(version=version)
    media = media_segment(version=version, extended=extended)
    fragment = tmp_path / 'chunk.mp4'
    fragment.write_bytes(initial + media)
    actual_init, actual_media = split_fragmented_mp4(str(fragment), timeline_offset=1.25)
    assert actual_init == initial
    assert len(actual_media) == len(media)
    assert decode_times(actual_media) == [125005, 60005]
    assert actual_media.endswith(b'encoded samples remain unchanged')


@pytest.mark.parametrize('offset', [-1, math.nan, math.inf])
def test_rejects_invalid_offsets(tmp_path, offset):
    with pytest.raises(ValueError, match='finite and non-negative'):
        split_fragmented_mp4(str(tmp_path / 'unused.mp4'), timeline_offset=offset)


def test_unknown_track_and_version_zero_overflow_fail_explicitly(tmp_path):
    fragment = tmp_path / 'chunk.mp4'
    fragment.write_bytes(init_segment() + media_segment(track_id=99))
    with pytest.raises(ValueError, match='matching init timescale'):
        split_fragmented_mp4(str(fragment), timeline_offset=1)
    fragment.write_bytes(init_segment() + media_segment(version=0, base=2**32 - 1))
    with pytest.raises(ValueError, match='field width'):
        split_fragmented_mp4(str(fragment), timeline_offset=1)


@pytest.mark.parametrize('suffix', [b'bad', struct.pack('>I4s', 1, b'mdat'), struct.pack('>I4s', 1000, b'mdat')])
def test_rejects_truncated_box_instead_of_publishing_partial_media(tmp_path, suffix):
    fragment = tmp_path / 'chunk.mp4'
    fragment.write_bytes(init_segment() + suffix)
    with pytest.raises(ValueError, match='Truncated'):
        split_fragmented_mp4(str(fragment), timeline_offset=1)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg/ffprobe unavailable')
def test_real_ffmpeg_chunks_have_absolute_decodable_timeline(tmp_path):
    video = tmp_path / 'video.mp4'
    audio = tmp_path / 'audio.wav'
    ff = FFmpegClient()
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                    'color=c=black:s=64x64:r=10:d=1', '-c:v', 'libx264', '-g', '10', '-bf', '0', str(video)], check=True)
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                    'sine=frequency=330:duration=1', '-ar', '48000', '-ac', '2', str(audio)], check=True)
    medias = []
    inits = []
    for chunk in range(2):
        fragment = tmp_path / f'chunk{chunk}.mp4'
        ff.mux_chunk_fmp4(str(video), str(audio), str(fragment))
        initial, media = split_fragmented_mp4(str(fragment), timeline_offset=float(chunk))
        inits.append(initial)
        medias.append(media)
    assert inits[0] == inits[1]
    assert decode_times(medias[0])[0] == 0
    assert decode_times(medias[1])[0] > 0
    assembled = tmp_path / 'assembled.mp4'
    assembled.write_bytes(inits[0] + b''.join(medias))
    packets = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                              'packet=pts_time', '-of', 'csv=p=0', str(assembled)], check=True, capture_output=True, text=True)
    times = [float(value.strip(',')) for value in packets.stdout.splitlines()]
    assert len(times) == 20
    assert all(right > left for left, right in zip(times, times[1:]))
    assert times[10] >= 1.0
