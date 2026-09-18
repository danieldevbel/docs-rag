"""Testes da avaliacao, incluindo o guarda de qualidade da CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from docsrag.chunking import ChunkConfig
from docsrag.corpus import load_corpus
from docsrag.evaluation import GoldenCase, evaluate, load_golden
from docsrag.retrieval import BM25Retriever

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
GOLDEN_PATH = Path(__file__).resolve().parents[1] / "eval" / "golden.json"

MIN_RECALL = 0.85
MIN_MRR = 0.70


@pytest.fixture(scope="module")
def retriever() -> BM25Retriever:
    return BM25Retriever(load_corpus(CORPUS_DIR))


def test_corpus_de_exemplo_e_indexavel(retriever):
    assert len(retriever) > 10


def test_conjunto_dourado_e_valido():
    cases = load_golden(GOLDEN_PATH)
    assert len(cases) >= 10
    assert all(case.relevant_ids for case in cases)


def test_ids_do_conjunto_dourado_existem_no_corpus():
    existentes = {c.chunk_id for c in load_corpus(CORPUS_DIR)}
    for case in load_golden(GOLDEN_PATH):
        faltando = set(case.relevant_ids) - existentes
        assert not faltando, f"ids inexistentes em '{case.question}': {faltando}"


def test_qualidade_da_recuperacao_nao_regride(retriever):
    """Guarda de qualidade: a CI falha se o recall ou o MRR cair."""
    report = evaluate(retriever, load_golden(GOLDEN_PATH), k=5)
    assert report.meets(MIN_RECALL, MIN_MRR), report.as_dict()


def test_relatorio_serializa_para_artefato(retriever):
    report = evaluate(retriever, load_golden(GOLDEN_PATH), k=5)
    data = report.as_dict()
    assert set(data) == {"cases", "k", "recall_at_k", "mrr"}


def test_caso_sem_trecho_relevante_e_rejeitado():
    with pytest.raises(ValueError, match="sem trecho relevante"):
        GoldenCase(question="p?", relevant_ids=[])


def test_caso_com_pergunta_vazia_e_rejeitado():
    with pytest.raises(ValueError, match="pergunta vazia"):
        GoldenCase(question="  ", relevant_ids=["x"])


def test_conjunto_vazio_e_rejeitado(retriever):
    with pytest.raises(ValueError, match="dourado vazio"):
        evaluate(retriever, [], k=5)


def test_k_invalido_e_rejeitado(retriever):
    with pytest.raises(ValueError, match="k deve ser positivo"):
        evaluate(retriever, load_golden(GOLDEN_PATH), k=0)


def test_recall_cai_com_k_menor(retriever):
    cases = load_golden(GOLDEN_PATH)
    assert (
        evaluate(retriever, cases, k=1).recall_at_k <= evaluate(retriever, cases, k=5).recall_at_k
    )


def test_fatiamento_agressivo_piora_a_metrica():
    """Demonstra que a metrica reage a mudanca de fatiamento, como deve."""
    picotado = BM25Retriever(
        load_corpus(CORPUS_DIR, ChunkConfig(max_chars=80, overlap_chars=0, min_chars=0))
    )
    cases = load_golden(GOLDEN_PATH)
    assert (
        evaluate(picotado, cases, k=5).mrr
        < evaluate(BM25Retriever(load_corpus(CORPUS_DIR)), cases, k=5).mrr
    )
