"""Interface de linha de comando."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from docsrag import __version__
from docsrag.chunking import ChunkConfig
from docsrag.corpus import load_corpus
from docsrag.evaluation import evaluate, load_golden
from docsrag.generation import EchoLLM, Generator, GeneratorConfig
from docsrag.retrieval import BM25Retriever
from docsrag.settings import get_settings

app = typer.Typer(
    add_completion=False,
    help="RAG sobre normativos institucionais.",
    no_args_is_help=True,
)


def _build_retriever(corpus_dir: Path) -> BM25Retriever:
    """Carrega o corpus e monta o indice lexico."""
    settings = get_settings()
    chunks = load_corpus(
        corpus_dir,
        ChunkConfig(
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
    )
    return BM25Retriever(chunks)


@app.command()
def version() -> None:
    """Mostra a versao instalada."""
    typer.echo(__version__)


@app.command()
def index(
    corpus: Path = typer.Option(Path("corpus"), help="Pasta dos normativos."),
) -> None:
    """Fatia o corpus e mostra a distribuicao dos trechos por documento."""
    retriever = _build_retriever(corpus)
    typer.echo(f"trechos indexados: {len(retriever)}")


@app.command()
def search(
    question: str = typer.Argument(..., help="Pergunta ou termo de busca."),
    corpus: Path = typer.Option(Path("corpus"), help="Pasta dos normativos."),
    k: int = typer.Option(5, min=1, max=50),
) -> None:
    """Mostra os trechos recuperados para uma pergunta."""
    for scored in _build_retriever(corpus).search(question, k=k):
        typer.echo(f"\n[{scored.chunk_id}] {scored.chunk.citation}  (score {scored.score:.3f})")
        typer.echo(scored.chunk.text[:220].replace("\n", " "))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pergunta."),
    corpus: Path = typer.Option(Path("corpus"), help="Pasta dos normativos."),
) -> None:
    """Responde a pergunta com o modelo falso, exercitando o fluxo completo."""
    settings = get_settings()
    retriever = _build_retriever(corpus)
    generator = Generator(
        EchoLLM(),
        GeneratorConfig(require_citation=settings.require_citation),
    )

    answer = generator.answer(question, retriever.search(question, k=settings.top_k))

    typer.echo(answer.text)
    if answer.citations:
        typer.echo("\nFontes:")
        for citation in answer.citations:
            typer.echo(f"  - {citation}")


@app.command(name="eval")
def run_eval(
    corpus: Path = typer.Option(Path("corpus"), help="Pasta dos normativos."),
    golden: Path = typer.Option(Path("eval/golden.json"), help="Conjunto dourado."),
    k: int = typer.Option(5, min=1, max=50),
    strict: bool = typer.Option(
        False, help="Sai com codigo 1 se o resultado ficar abaixo dos limiares."
    ),
) -> None:
    """Mede recall@k e MRR da recuperacao sobre o conjunto dourado."""
    settings = get_settings()
    report = evaluate(_build_retriever(corpus), load_golden(golden), k=k)

    typer.echo(json.dumps(report.as_dict(), indent=2))

    if strict and not report.meets(settings.min_recall_at_k, settings.min_mrr):
        typer.echo(
            f"\nabaixo do limiar: recall>={settings.min_recall_at_k} e mrr>={settings.min_mrr}",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
