"""Testes da recuperacao."""

from __future__ import annotations

import numpy as np
import pytest

from docsrag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    normalize_text,
    reciprocal_rank_fusion,
    tokenize,
)
from docsrag.types import Chunk, Citation, ScoredChunk


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, text=text, citation=Citation("d", "Doc", cid))


CHUNKS = [
    chunk("c1", "E dispensavel o procedimento licitatorio quando o valor for inferior."),
    chunk("c2", "A estimativa de preco sera obtida por consulta de mercado a tres fornecedores."),
    chunk("c3", "Incidentes de seguranca serao comunicados em vinte e quatro horas."),
    chunk("c4", "O acesso a sistemas segue o principio do menor privilegio."),
]


def test_normalize_remove_acento():
    assert normalize_text("Licitação Pública") == "licitacao publica"


def test_tokenize_descarta_stopwords():
    assert "de" not in tokenize("consulta de mercado")
    assert "consulta" in tokenize("consulta de mercado")


def test_tokenize_pode_manter_stopwords():
    assert "de" in tokenize("consulta de mercado", drop_stopwords=False)


def test_indice_vazio_e_rejeitado():
    with pytest.raises(ValueError, match="ao menos um trecho"):
        BM25Retriever([])


def test_busca_encontra_o_trecho_certo():
    result = BM25Retriever(CHUNKS).search("dispensa licitatorio valor", k=2)
    assert result[0].chunk_id == "c1"


def test_busca_ignora_acento_da_consulta():
    result = BM25Retriever(CHUNKS).search("incidentes de segurança", k=1)
    assert result[0].chunk_id == "c3"


def test_consulta_sem_termo_util_devolve_vazio():
    assert BM25Retriever(CHUNKS).search("de a o os", k=3) == []


def test_termo_inexistente_devolve_vazio():
    assert BM25Retriever(CHUNKS).search("zebra quantica", k=3) == []


def test_respeita_o_limite_k():
    assert len(BM25Retriever(CHUNKS).search("procedimento seguranca acesso preco", k=2)) <= 2


def test_score_decresce_na_ordem():
    result = BM25Retriever(CHUNKS).search("seguranca acesso sistemas", k=4)
    scores = [s.score for s in result]
    assert scores == sorted(scores, reverse=True)


def test_dense_exige_mesma_quantidade():
    with pytest.raises(ValueError, match="incompativel"):
        DenseRetriever(CHUNKS, np.eye(2, dtype=np.float32))


def test_dense_rejeita_vetor_nulo():
    embeddings = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="vetor nulo"):
        DenseRetriever(CHUNKS, embeddings)


def test_dense_encontra_o_vetor_mais_proximo():
    embeddings = np.eye(4, dtype=np.float32)
    retriever = DenseRetriever(CHUNKS, embeddings)
    result = retriever.search_vector(np.array([0, 0, 1, 0], dtype=np.float32), k=1)
    assert result[0].chunk_id == "c3"


def test_dense_rejeita_consulta_nula():
    with pytest.raises(ValueError, match="consulta nulo"):
        DenseRetriever(CHUNKS, np.eye(4, dtype=np.float32)).search_vector(
            np.zeros(4, dtype=np.float32)
        )


def test_fusao_premia_quem_aparece_bem_nos_dois_rankings():
    lexico = [ScoredChunk(CHUNKS[0], 9.0), ScoredChunk(CHUNKS[1], 8.0)]
    denso = [ScoredChunk(CHUNKS[1], 0.9), ScoredChunk(CHUNKS[2], 0.8)]
    fused = reciprocal_rank_fusion([lexico, denso], k=3)
    assert fused[0].chunk_id == "c2"


def test_fusao_nao_duplica_trecho():
    ranking = [ScoredChunk(CHUNKS[0], 1.0)]
    fused = reciprocal_rank_fusion([ranking, ranking], k=5)
    assert len(fused) == 1


def test_fusao_rejeita_constante_invalida():
    with pytest.raises(ValueError, match="constant"):
        reciprocal_rank_fusion([], k=1, constant=0)
