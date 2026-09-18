"""Recuperacao lexica, densa e hibrida.

O BM25 e implementado aqui em vez de importado para que o comportamento seja
inspecionavel e testavel sem servico externo. Em normativo, a busca lexica
costuma ganhar da densa: o usuario procura o termo exato da norma
('dispensa de licitacao'), nao uma parafrase.

A fusao hibrida usa Reciprocal Rank Fusion, que combina rankings sem exigir
que as pontuacoes estejam na mesma escala, o que e justamente o problema de
somar BM25 com similaridade de cosseno.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from docsrag.types import Chunk, ScoredChunk

STOPWORDS = frozenset(
    [
        "a",
        "as",
        "o",
        "os",
        "um",
        "uma",
        "uns",
        "umas",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "por",
        "para",
        "com",
        "sem",
        "sob",
        "sobre",
        "e",
        "ou",
        "que",
        "se",
        "ao",
        "aos",
        "à",
        "às",
        "pelo",
        "pela",
        "pelos",
        "pelas",
        "seu",
        "sua",
        "seus",
        "suas",
        "este",
        "esta",
        "estes",
        "estas",
        "esse",
        "essa",
        "esses",
        "essas",
        "aquele",
        "aquela",
        "é",
        "ser",
        "são",
        "foi",
        "como",
        "mais",
        "menos",
        "ja",
        "nao",
        "não",
    ]
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Minusculiza e remove acentos, para que 'licitação' case com 'licitacao'."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Quebra o texto em termos indexaveis."""
    tokens = TOKEN_PATTERN.findall(normalize_text(text))
    if not drop_stopwords:
        return tokens
    normalized_stop = {normalize_text(w) for w in STOPWORDS}
    return [t for t in tokens if t not in normalized_stop]


class Retriever(Protocol):
    """Contrato de um recuperador de trechos."""

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Devolve os k trechos mais relevantes para a consulta."""
        ...


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Parametros do BM25."""

    k1: float = 1.5
    """Saturacao da frequencia do termo. Valores maiores premiam repeticao."""

    b: float = 0.75
    """Peso da normalizacao por tamanho do documento."""


class BM25Retriever:
    """Busca lexica por BM25 Okapi."""

    def __init__(self, chunks: list[Chunk], config: BM25Config | None = None) -> None:
        if not chunks:
            raise ValueError("o indice precisa de ao menos um trecho")

        self._config = config or BM25Config()
        self._chunks = chunks
        self._docs = [tokenize(c.text) for c in chunks]
        self._lengths = np.array([len(d) for d in self._docs], dtype=np.float32)
        self._avg_length = float(self._lengths.mean()) or 1.0
        self._term_freqs = [Counter(d) for d in self._docs]

        doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            doc_freq.update(set(doc))

        n = len(chunks)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Devolve os k trechos com maior pontuacao BM25."""
        terms = tokenize(query)
        if not terms:
            return []

        k1, b = self._config.k1, self._config.b
        scores = np.zeros(len(self._chunks), dtype=np.float32)

        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self._term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                norm = 1 - b + b * (self._lengths[i] / self._avg_length)
                scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * norm)

        return self._top_k(scores, k)

    def _top_k(self, scores: np.ndarray, k: int) -> list[ScoredChunk]:
        """Seleciona os k maiores, descartando pontuacao nula."""
        order = np.argsort(-scores)[: max(k, 0)]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in order
            if scores[i] > 0
        ]


Embedder = "typing.Callable[[str], np.ndarray]"


class DenseRetriever:
    """Busca densa por similaridade de cosseno sobre vetores pre-calculados."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"quantidade incompativel: {len(chunks)} trechos, {embeddings.shape[0]} vetores"
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("ha vetor nulo entre os embeddings")

        self._chunks = chunks
        self._matrix = (embeddings / norms).astype(np.float32)

    def search_vector(self, query_vector: np.ndarray, k: int = 5) -> list[ScoredChunk]:
        """Busca pelos trechos mais proximos de um vetor de consulta."""
        norm = float(np.linalg.norm(query_vector))
        if norm == 0.0:
            raise ValueError("vetor de consulta nulo")

        scores = self._matrix @ (query_vector / norm).astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in order]


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]],
    k: int = 5,
    constant: int = 60,
) -> list[ScoredChunk]:
    """Funde varios rankings pela soma de 1 / (constante + posicao).

    Nao exige que as pontuacoes de origem estejam na mesma escala, que e a
    razao de nao somar BM25 com cosseno diretamente. A constante amortece o
    peso das primeiras posicoes; 60 e o valor da literatura original.
    """
    if constant <= 0:
        raise ValueError("constant deve ser positivo")

    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            fused[scored.chunk_id] = fused.get(scored.chunk_id, 0.0) + 1.0 / (constant + rank)
            by_id[scored.chunk_id] = scored.chunk

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [ScoredChunk(chunk=by_id[cid], score=score) for cid, score in ordered]
