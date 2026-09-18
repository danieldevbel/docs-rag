"""Fatiamento de documentos normativos.

Fatiar por contagem fixa de caracteres corta artigo ao meio e destroi a
citacao. Aqui o fatiamento segue a estrutura do texto: primeiro tenta quebrar
em artigos, e so recorre a janela deslizante quando o trecho e grande demais ou
quando o documento nao tem estrutura reconhecivel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docsrag.types import Chunk, Citation

ARTICLE_PATTERN = re.compile(
    r"^\s*(Art\.?\s*\d+\.?[º°o]?|Artigo\s+\d+\.?[º°o]?|"
    r"Par[aá]grafo\s+[Uu]nico|§\s*\d+[º°o]?)",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Parametros do fatiamento."""

    max_chars: int = 1200
    """Tamanho maximo de um trecho antes de recorrer a janela deslizante."""

    overlap_chars: int = 150
    """Sobreposicao entre janelas, para nao perder frase cortada na borda."""

    min_chars: int = 60
    """Trechos menores que isso sao anexados ao anterior em vez de virar trecho."""

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars deve ser positivo")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("overlap_chars deve estar em [0, max_chars)")
        if self.min_chars < 0:
            raise ValueError("min_chars nao pode ser negativo")


def split_articles(text: str) -> list[tuple[str | None, str]]:
    """Quebra o texto nos marcadores de artigo.

    Returns:
        Lista de pares (rotulo do artigo, texto). O rotulo e `None` para o
        preambulo que antecede o primeiro artigo.
    """
    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        stripped = text.strip()
        return [(None, stripped)] if stripped else []

    parts: list[tuple[str | None, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        parts.append((None, preamble))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        if body:
            parts.append((_normalize_label(match.group(1)), body))

    return parts


def _normalize_label(raw: str) -> str:
    """Normaliza o rotulo do artigo para uso em citacao."""
    label = " ".join(raw.split()).rstrip(".")
    return label.replace("Art ", "Art. ").replace("Artigo", "Art.")


def _windows(text: str, config: ChunkConfig) -> list[str]:
    """Quebra um texto longo em janelas com sobreposicao."""
    if len(text) <= config.max_chars:
        return [text]

    step = config.max_chars - config.overlap_chars
    return [
        text[i : i + config.max_chars].strip()
        for i in range(0, len(text), step)
        if text[i : i + config.max_chars].strip()
    ]


def chunk_document(
    document_id: str,
    title: str,
    text: str,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Fatia um documento normativo em trechos citaveis."""
    cfg = config or ChunkConfig()
    chunks: list[Chunk] = []
    pending: tuple[str | None, str] | None = None

    for label, body in split_articles(text):
        if pending is not None:
            label, body = pending[0], f"{pending[1]}\n{body}"
            pending = None

        if len(body) < cfg.min_chars:
            pending = (label, body)
            continue

        for window in _windows(body, cfg):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}#{len(chunks):04d}",
                    text=window,
                    citation=Citation(document_id=document_id, title=title, article=label),
                    position=len(chunks),
                )
            )

    if pending is not None and pending[1].strip():
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}#{len(chunks):04d}",
                text=pending[1],
                citation=Citation(document_id=document_id, title=title, article=pending[0]),
                position=len(chunks),
            )
        )

    return chunks
