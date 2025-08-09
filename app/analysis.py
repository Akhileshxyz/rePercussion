from __future__ import annotations

from typing import Iterable, List, Tuple


def analyze_audio_features(audio_file_paths: Iterable[str]) -> List[dict]:
    """Analyze audio files using librosa and return a list of feature dicts.

    This is a placeholder implementation that defers heavy imports until used.
    """
    import librosa  # lazy import to avoid heavy startup cost

    features: List[dict] = []
    for path in audio_file_paths:
        try:
            signal, sr = librosa.load(path, mono=True)
            tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).mean(axis=1).tolist()
            features.append({"path": path, "tempo": float(tempo), "mfcc_mean": mfcc})
        except Exception as exc:
            features.append({"path": path, "error": str(exc)})
    return features


def compute_text_embeddings(texts: Iterable[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[List[List[float]], str]:
    """Compute sentence embeddings for a collection of texts.

    Returns a tuple of (embeddings, model_name_used).
    """
    from sentence_transformers import SentenceTransformer  # lazy import

    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(texts), convert_to_numpy=False)
    return [emb.tolist() for emb in embeddings], model_name


