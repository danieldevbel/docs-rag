"""API HTTP do servico."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from docsrag import __version__
from docsrag.generation import Generator
from docsrag.retrieval import Retriever


class AskRequest(BaseModel):
    """Corpo da requisicao de pergunta."""

    question: str = Field(min_length=3, max_length=1000)


def create_app(retriever: Retriever, generator: Generator, top_k: int = 5) -> Any:
    """Monta a aplicacao FastAPI sobre um recuperador e um gerador ja prontos."""
    try:
        from fastapi import FastAPI, Query
    except ImportError as exc:  # pragma: no cover - depende do extra
        raise RuntimeError("a API requer o extra 'api': pip install 'docs-rag[api]'") from exc

    app = FastAPI(
        title="docs-rag",
        version=__version__,
        description="RAG sobre normativos institucionais, com citacao obrigatoria.",
    )

    @app.get("/health", tags=["operacao"])
    def health() -> dict[str, str]:
        """Verificacao de saude."""
        return {"status": "ok", "version": __version__}

    @app.get("/search", tags=["busca"])
    def search(
        q: str = Query(min_length=2),
        k: int = Query(default=top_k, ge=1, le=50),
    ) -> list[dict[str, Any]]:
        """Devolve os trechos recuperados, sem passar por modelo de linguagem."""
        return [
            {
                "chunk_id": s.chunk_id,
                "score": round(s.score, 4),
                "citation": str(s.chunk.citation),
                "text": s.chunk.text,
            }
            for s in retriever.search(q, k=k)
        ]

    @app.post("/ask", tags=["resposta"])
    def ask(body: AskRequest) -> dict[str, Any]:
        """Responde a pergunta, sempre indicando se a resposta esta sustentada."""
        answer = generator.answer(body.question, retriever.search(body.question, k=top_k))
        return {
            "answer": answer.text,
            "grounded": answer.grounded,
            "citations": [str(c) for c in answer.citations],
            "chunk_ids": [s.chunk_id for s in answer.sources],
        }

    return app
