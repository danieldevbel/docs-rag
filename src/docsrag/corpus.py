"""Ingestao do corpus a partir do sistema de arquivos."""

from __future__ import annotations

from pathlib import Path

from docsrag.chunking import ChunkConfig, chunk_document
from docsrag.types import Chunk


def title_from_text(text: str, fallback: str) -> str:
    """Usa a primeira linha nao vazia do documento como titulo."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return " ".join(w.capitalize() if w.isupper() else w for w in stripped.split())
    return fallback


def load_corpus(directory: Path, config: ChunkConfig | None = None) -> list[Chunk]:
    """Le todos os arquivos .txt de uma pasta e devolve os trechos indexaveis.

    Raises:
        FileNotFoundError: se a pasta nao existir.
        ValueError: se nenhum documento legivel for encontrado.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"pasta de corpus nao encontrada: {directory}")

    chunks: list[Chunk] = []

    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        chunks.extend(
            chunk_document(
                document_id=path.stem,
                title=title_from_text(text, path.stem),
                text=text,
                config=config,
            )
        )

    if not chunks:
        raise ValueError(f"nenhum documento .txt legivel em {directory}")

    return chunks
