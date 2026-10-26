"""
vectorize.py - CLEAN_TITLE 및 검색 질의를 512차원 Feature Hashing 임베딩 벡터(title_vec)로 변환하고
sqlite-vec 호환 BLOB으로 직렬화하는 모듈
"""

from __future__ import annotations

import hashlib
import numpy as np
import sqlite_vec

from preprocessing import TOKEN_RE, preprocess_title

TITLE_VEC_DIM = 512


def text_to_vector(text: str, dim: int = TITLE_VEC_DIM) -> np.ndarray:
    """
    CLEAN_TITLE 또는 검색 질의를 단어 + 음절 바이그램/트라이그램 Feature Hashing 기반
    L2 정규화 float32 임베딩 벡터(dim=512)로 고속 변환합니다.
    """
    vec = np.zeros(dim, dtype=np.float32)
    clean = preprocess_title(text) or text
    tokens = TOKEN_RE.findall(clean)
    if not tokens:
        tokens = TOKEN_RE.findall(text)

    features: list[tuple[str, float]] = []
    for tok in tokens:
        features.append((f"w:{tok}", 2.2))
        for n in (2, 3):
            if len(tok) >= n:
                for i in range(len(tok) - n + 1):
                    features.append((f"c{n}:{tok[i:i+n]}", 1.0))

    for feat, weight in features:
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def serialize_vec(vec: np.ndarray) -> bytes:
    """numpy float32 배열을 sqlite-vec 호환 직렬화 BLOB으로 변환합니다."""
    return sqlite_vec.serialize_float32(vec.astype(np.float32).tolist())


__all__ = [
    "TITLE_VEC_DIM",
    "text_to_vector",
    "serialize_vec",
]
