"""Speaker diarization without gated pyannote: ECAPA embeddings + clustering.

Assigns SPEAKER_00, SPEAKER_01, ... to whisper words by time overlap.
"""
from __future__ import annotations

import logging
import os
from bisect import bisect_left

import numpy as np

from langswap.model_config import MODEL_WEIGHTS_DIR

logger = logging.getLogger(__name__)

def _classifier(device: str):
    from langswap.model_pool import get_or_create

    run_device = "cuda" if (device or "").startswith("cuda") else "cpu"
    return get_or_create(("diarization", "ecapa", run_device), lambda: _load_classifier(run_device))


def _load_classifier(run_device: str):
    os.environ.setdefault("HF_HOME", MODEL_WEIGHTS_DIR)
    from speechbrain.inference.speaker import EncoderClassifier

    savedir = os.path.join(MODEL_WEIGHTS_DIR, "ecapa_voxceleb")
    logger.info("Loading ECAPA speaker encoder device=%s", run_device)
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": run_device},
    )
def _cluster(embs: np.ndarray, num_speakers: int | None = None) -> np.ndarray:
    from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
    from scipy.spatial.distance import pdist

    if num_speakers is not None and not 1 <= num_speakers <= len(embs):
        raise ValueError("num_speakers must be positive and no greater than the number of voiced windows")
    if len(embs) < 2 or num_speakers == 1:
        return np.zeros(len(embs), dtype=int)
    # cosine distance on L2-normalized embeddings
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    x = embs / norms
    d = pdist(x, metric="cosine")
    z = linkage(d, method="average")
    labels = (fcluster(z, t=0.42, criterion="distance") if num_speakers is None
              else cut_tree(z, n_clusters=[num_speakers]).ravel())
    # Canonical IDs follow first appearance; a returning person keeps the same ID.
    ids = {}
    return np.array([ids.setdefault(int(label), len(ids)) for label in labels])


def _anchor_centroids(embeddings, complete_utterances, num_speakers=None):
    """Cluster identity from sustained speech, excluding isolated mixed windows.

    A short *complete* utterance remains evidence for a person. An isolated
    window inside a longer utterance can straddle two people; it must not create
    a new identity without a second supporting window. Explicit speaker counts
    retain every requested cluster instead of applying this automatic heuristic.
    """
    labels = _cluster(embeddings, num_speakers=num_speakers)
    normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    centers, support = [], []
    for label in np.unique(labels):
        members = labels == label
        if (num_speakers is not None or members.sum() >= 2
                or any(whole for whole, member in zip(complete_utterances, members) if member)):
            center = normalized[members].mean(axis=0)
            centers.append(center / (np.linalg.norm(center) + 1e-8))
            support.append(int(members.sum()))
    if not centers:
        raise ValueError("No sustained or complete utterance supports a speaker identity")
    centers = np.stack(centers)
    if num_speakers is None and len(centers) > 1:
        # Average linkage over individual windows can split a returning voice:
        # its phonetic variants disagree even though their sustained prototypes
        # meet the same cosine threshold. Consolidate those global prototypes,
        # retaining a distinct short speaker when its prototype is dissimilar.
        merged = _cluster(centers)
        combined = []
        for label in np.unique(merged):
            members = merged == label
            if members.sum() == 1:
                combined.append(centers[members][0])
                continue
            center = np.average(centers[members], axis=0, weights=np.asarray(support)[members])
            combined.append(center / (np.linalg.norm(center) + 1e-8))
        centers = np.stack(combined)
    return centers


def assign_word_speakers(
    words: list[dict],
    audio_16k: np.ndarray,
    device: str = "cuda",
    win_s: float = 1.25,
    hop_s: float = 0.4,
    num_speakers: int | None = None,
) -> list[dict]:
    """Mutates words in place with a 'speaker' key. Returns words."""
    if not words:
        return words
    if win_s <= 0 or hop_s <= 0:
        raise ValueError("Diarization window and hop must be positive")
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("num_speakers must be positive")
    sr = 16000
    wav = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    if wav.size < int(0.6 * sr) or num_speakers == 1:
        if num_speakers is not None and num_speakers > 1:
            raise ValueError("Audio is too short for the requested speaker count")
        for w in words:
            w["speaker"] = "SPEAKER_00"
        return words

    # Whisper gives us speech intervals. RMS alone includes music-only windows,
    # whose unrelated embeddings would create spurious speakers.
    speech = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if speech and start <= speech[-1][1] + 0.25:
            speech[-1] = (speech[-1][0], max(end, speech[-1][1]))
        else:
            speech.append((start, end))
    clf = _classifier(device)
    import torch

    # 1.25 s was sufficient to locate a change but not to identify a person:
    # ECAPA fragmented one real 50 s clip into 45 phonetic/mixed-window labels.
    # Four-second anchors supply identity; short local windows locate the turns.
    encoded = {}

    def encode(start, end):
        key = start, end
        if key in encoded:
            return encoded[key]
        sl = wav[start:end]
        rms = float(np.sqrt(np.mean(sl * sl) + 1e-12))
        if rms < 0.008:
            return None
        t = torch.from_numpy(sl.copy()).unsqueeze(0)
        with torch.no_grad():
            e = clf.encode_batch(t)
        encoded[key] = e.squeeze().detach().cpu().numpy().reshape(-1)
        return encoded[key]

    def windows(window_seconds, hop_seconds):
        width, step = max(1, round(window_seconds * sr)), max(1, round(hop_seconds * sr))
        for start, end in speech:
            left, right = max(0, round(start * sr)), min(wav.size, round(end * sr))
            if right <= left:
                continue
            win = min(width, right - left)
            starts = list(range(left, right - win + 1, step))
            if starts[-1] != right - win:
                starts.append(right - win)
            for first in starts:
                embedding = encode(first, first + win)
                if embedding is not None:
                    yield embedding, (first / sr, (first + win) / sr), win == right - left

    anchors = list(windows(4.0, 1.0))
    if not anchors:
        raise ValueError("No voiced diarization windows matched the transcript")
    centroids = _anchor_centroids(np.stack([e for e, _, _ in anchors]),
                                  [whole for _, _, whole in anchors], num_speakers)
    local = list(windows(win_s, hop_s))
    embs = np.stack([e for e, _, _ in local])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    labels = np.argmax(embs @ centroids.T, axis=1)
    times = [time for _, time, _ in local]
    logger.info("embed diarize: %s anchors, %s local windows, %s speaker identities",
                len(anchors), len(local), len(centroids))
    centers = [(a + b) / 2 for a, b in times]

    def speaker_at(t: float) -> str:
        # Overlapping windows cover ~1.25 s. Taking the first containing window
        # delays every speaker change; select the closest center instead.
        index = min(bisect_left(centers, t), len(centers) - 1)
        if index > 0 and abs(t - centers[index - 1]) <= abs(t - centers[index]):
            index -= 1
        return f"SPEAKER_{int(labels[index]):02d}"

    canonical = {}
    for w in words:
        mid = (float(w.get("start", 0.0)) + float(w.get("end", 0.0))) / 2.0
        label = speaker_at(mid)
        w["speaker"] = f"SPEAKER_{canonical.setdefault(label, len(canonical)):02d}"
    return words
