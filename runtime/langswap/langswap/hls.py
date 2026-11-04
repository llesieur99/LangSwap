"""HLS / fMP4 helpers for streaming dubbing (Phase 0).

This module is pure-Python and has no ML / GPU / S3 dependencies, so it can be
unit-tested on its own against a synthetic clip.  Two pieces:

* ``split_fragmented_mp4`` — split a fragmented MP4 produced by ffmpeg
  (``ftyp + moov + (moof + mdat)+``) into the CMAF **init segment**
  (``ftyp + moov``, written once and referenced by ``#EXT-X-MAP``) and the
  **media segment** (``moof + mdat ...``, an independently-decodable ``.m4s``).

* ``HlsManifest`` — an append-only HLS **EVENT** playlist writer.  Segments are
  added as they are produced; ``finalize()`` writes ``#EXT-X-ENDLIST`` which
  flips the player from live to a fully-seekable VOD asset.

See docs/streaming_dubbing_design.md §2.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# fMP4 box parsing
# ---------------------------------------------------------------------------

def _iter_boxes(data: bytes):
    """Yield ``(box_type, start, end)`` for each top-level ISO-BMFF box.

    Handles the 32-bit size, the ``size == 1`` 64-bit largesize escape, and the
    ``size == 0`` "to end of file" case.  Only top-level boxes are walked, which
    is all we need to find the ftyp/moov/moof boundaries.
    """
    offset = 0
    n = len(data)
    while offset + 8 <= n:
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        box_type = data[offset + 4:offset + 8].decode("latin-1")
        header = 8
        if size == 1:
            # 64-bit largesize in the 8 bytes following the type
            if offset + 16 > n:
                raise ValueError('Truncated MP4 extended box header')
            size = struct.unpack(">Q", data[offset + 8:offset + 16])[0]
            header = 16
        elif size == 0:
            size = n - offset  # extends to EOF
        if size < header or offset + size > n:
            raise ValueError('Truncated or malformed MP4 box')
        yield box_type, offset, offset + size
        offset += size
    if offset != n:
        raise ValueError('Truncated MP4 box header')


def _children(data, start, end):
    for kind, first, last in _iter_boxes(data[start:end]):
        first += start
        last += start
        header = 16 if struct.unpack_from('>I', data, first)[0] == 1 else 8
        yield kind, first + header, last


def _shift_decode_times(init_bytes, media_bytes, timeline_offset):
    """Keep box sizes/media payload intact while placing each track on its timeline."""
    timescales = {}
    for kind, moov, moov_end in _children(init_bytes, 0, len(init_bytes)):
        if kind != 'moov':
            continue
        for kind, trak, trak_end in _children(init_bytes, moov, moov_end):
            if kind != 'trak':
                continue
            track_id = None
            timescale = None
            for kind, payload, end in _children(init_bytes, trak, trak_end):
                if kind == 'tkhd':
                    if payload >= end or init_bytes[payload] not in (0, 1):
                        raise ValueError('Invalid MP4 track header version')
                    offset = 20 if init_bytes[payload] == 1 else 12
                    if payload + offset + 4 > end:
                        raise ValueError('Truncated MP4 track header')
                    track_id = struct.unpack_from('>I', init_bytes, payload + offset)[0]
                elif kind == 'mdia':
                    for kind, header, header_end in _children(init_bytes, payload, end):
                        if kind == 'mdhd':
                            if header >= header_end or init_bytes[header] not in (0, 1):
                                raise ValueError('Invalid MP4 media header version')
                            offset = 20 if init_bytes[header] == 1 else 12
                            if header + offset + 4 > header_end:
                                raise ValueError('Truncated MP4 media header')
                            timescale = struct.unpack_from('>I', init_bytes, header + offset)[0]
            if track_id is None or not timescale:
                raise ValueError('Missing MP4 track ID or timescale')
            timescales[track_id] = timescale

    media = bytearray(media_bytes)
    shifted = 0
    for kind, moof, moof_end in _children(media, 0, len(media)):
        if kind != 'moof':
            continue
        for kind, traf, traf_end in _children(media, moof, moof_end):
            if kind != 'traf':
                continue
            children = list(_children(media, traf, traf_end))
            track_id = None
            for kind, payload, end in children:
                if kind == 'tfhd':
                    if payload + 8 > end:
                        raise ValueError('Truncated MP4 fragment track header')
                    track_id = struct.unpack_from('>I', media, payload + 4)[0]
            if track_id not in timescales:
                raise ValueError('Fragment track has no matching init timescale')
            track_shifted = False
            for kind, payload, end in children:
                if kind == 'tfdt':
                    if payload + 4 > end:
                        raise ValueError('Truncated MP4 decode time header')
                    version = media[payload]
                    if version not in (0, 1):
                        raise ValueError('Unsupported MP4 decode time version')
                    fmt = '>Q' if version == 1 else '>I'
                    width = 8 if version == 1 else 4
                    if payload + 4 + width > end:
                        raise ValueError('Truncated MP4 decode time')
                    base = struct.unpack_from(fmt, media, payload + 4)[0]
                    absolute = base + round(timeline_offset * timescales[track_id])
                    if absolute >= 1 << (8 * width):
                        raise ValueError('MP4 decode time exceeds its field width')
                    struct.pack_into(fmt, media, payload + 4, absolute)
                    shifted += 1
                    track_shifted = True
            if not track_shifted:
                raise ValueError('Fragment track has no decode timestamp')
    if not shifted:
        raise ValueError('Fragment has no decode timestamps to place on timeline')
    return bytes(media)


def split_fragmented_mp4(frag_path: str, timeline_offset: float = 0.0) -> Tuple[bytes, bytes]:
    """Split a fragmented MP4 into ``(init_bytes, media_bytes)``.

    ``init`` = ``ftyp`` + ``moov`` (CMAF init segment).
    ``media`` = everything after ``moov`` (the ``moof``/``mdat`` fragments),
    with any trailing ``mfra`` random-access index dropped — it's optional and
    only bloats every media segment.
    ``timeline_offset`` places independently muxed chunks on the absolute HLS
    timeline in seconds, using each track's own timescale from the init segment.
    """
    if not math.isfinite(timeline_offset) or timeline_offset < 0:
        raise ValueError('HLS timeline offset must be finite and non-negative')
    with open(frag_path, "rb") as f:
        data = f.read()

    init_end = None
    media_start = None
    media_end = len(data)
    for box_type, start, end in _iter_boxes(data):
        if box_type == "moov":
            init_end = end
            media_start = end
        elif box_type == "mfra":
            media_end = min(media_end, start)

    if init_end is None or media_start is None:
        raise ValueError(
            f"{frag_path} is not a fragmented MP4 (no moov box found). "
            "Ensure ffmpeg ran with -movflags +empty_moov+frag_keyframe."
        )

    init_bytes = data[:init_end]
    media_bytes = data[media_start:media_end]
    if timeline_offset:
        media_bytes = _shift_decode_times(init_bytes, media_bytes, timeline_offset)
    return init_bytes, media_bytes


# ---------------------------------------------------------------------------
# HLS EVENT playlist
# ---------------------------------------------------------------------------

@dataclass
class HlsManifest:
    """Append-only HLS EVENT playlist writer (fMP4 segments).

    The playlist is rewritten in full on every append — HLS requires
    ``#EXT-X-TARGETDURATION`` to be >= the longest segment, so we recompute the
    header each time.  This is cheap (a few hundred bytes) and keeps the file
    valid for a player polling it mid-production.
    """

    playlist_path: str
    init_uri: str = "init.mp4"
    version: int = 7
    segments: List[Tuple[str, float]] = field(default_factory=list)  # (uri, duration_s)
    _finalized: bool = False

    def add_segment(self, uri: str, duration: float) -> None:
        self.segments.append((uri, float(duration)))
        self._write()

    def finalize(self) -> None:
        """Append #EXT-X-ENDLIST — the dub is complete, asset is now seekable."""
        self._finalized = True
        self._write()

    def _render(self) -> str:
        target = max((d for _, d in self.segments), default=1.0)
        lines = [
            "#EXTM3U",
            f"#EXT-X-VERSION:{self.version}",
            f"#EXT-X-TARGETDURATION:{int(math.ceil(target))}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:EVENT",
            "#EXT-X-INDEPENDENT-SEGMENTS",
            f'#EXT-X-MAP:URI="{self.init_uri}"',
        ]
        for uri, duration in self.segments:
            lines.append(f"#EXTINF:{duration:.3f},")
            lines.append(uri)
        if self._finalized:
            lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines) + "\n"

    def _write(self) -> None:
        tmp = self.playlist_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self._render())
        os.replace(tmp, self.playlist_path)  # atomic for concurrent pollers
