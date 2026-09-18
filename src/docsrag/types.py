"""Tipos de dominio.

Um `Chunk` de normativo nao e um pedaco arbitrario de texto: ele carrega a
norma e o artigo de onde saiu. Sem isso nao ha como citar, e uma resposta sem
citacao nao serve para decisao administrativa.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Citation:
    """Referencia normativa de um trecho."""

    document_id: str
    """Identificador do documento, por exemplo 'res-1593-2024'."""

    title: str
    """Titulo legivel, por exemplo 'Resolucao Sesc 1.593/2024'."""

    article: str | None = None
    """Artigo, inciso ou secao, quando identificavel."""

    def __str__(self) -> str:
        return f"{self.title}, {self.article}" if self.article else self.title


@dataclass(frozen=True, slots=True)
class Chunk:
    """Trecho indexavel de um documento, com sua referencia normativa."""

    chunk_id: str
    text: str
    citation: Citation
    position: int = 0
    """Ordem do trecho dentro do documento, usada para reconstruir contexto."""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id nao pode ser vazio")
        if not self.text.strip():
            raise ValueError("chunk sem texto")


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """Trecho recuperado com a pontuacao que o trouxe."""

    chunk: Chunk
    score: float

    @property
    def chunk_id(self) -> str:
        """Atalho para o identificador do trecho."""
        return self.chunk.chunk_id


@dataclass(frozen=True, slots=True)
class Answer:
    """Resposta gerada, com os trechos que a sustentam."""

    text: str
    sources: list[ScoredChunk] = field(default_factory=list)
    grounded: bool = True
    """Falso quando a resposta nao pode ser sustentada pelos trechos recuperados."""

    @property
    def citations(self) -> list[Citation]:
        """Referencias normativas unicas, na ordem de relevancia."""
        seen: set[str] = set()
        result: list[Citation] = []
        for scored in self.sources:
            key = str(scored.chunk.citation)
            if key not in seen:
                seen.add(key)
                result.append(scored.chunk.citation)
        return result
