"""Testes da API HTTP."""

from __future__ import annotations

from pathlib import Path

import pytest

from docsrag.api import create_app
from docsrag.corpus import load_corpus
from docsrag.generation import EchoLLM, Generator
from docsrag.retrieval import BM25Retriever

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="module")
def client() -> TestClient:
    retriever = BM25Retriever(load_corpus(CORPUS_DIR))
    return TestClient(create_app(retriever, Generator(EchoLLM())))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_search_devolve_citacao(client):
    body = client.get("/search", params={"q": "dispensa de licitacao", "k": 3}).json()
    assert 1 <= len(body) <= 3
    assert all("citation" in item for item in body)


def test_search_rejeita_consulta_curta(client):
    assert client.get("/search", params={"q": "a"}).status_code == 422


def test_ask_devolve_sustentacao_e_fontes(client):
    body = client.post("/ask", json={"question": "Qual o prazo de analise da requisicao?"}).json()
    assert body["grounded"] is True
    assert body["citations"]
    assert body["chunk_ids"]


def test_ask_sem_base_normativa_admite_que_nao_sabe(client):
    body = client.post("/ask", json={"question": "Qual a receita de bolo de cenoura?"}).json()
    assert body["grounded"] is False
    assert body["citations"] == []


def test_ask_rejeita_pergunta_curta(client):
    assert client.post("/ask", json={"question": "a"}).status_code == 422
