"""Testes do fatiamento por artigo."""

from __future__ import annotations

import pytest

from docsrag.chunking import ChunkConfig, chunk_document, split_articles

NORMA = """RESOLUCAO 001/2025

Preambulo com a ementa da norma, suficientemente longo para virar um trecho
proprio e nao ser anexado ao artigo seguinte pelo tamanho minimo.

Art. 1o Primeiro artigo com texto suficiente para ultrapassar o tamanho minimo
exigido pela configuracao padrao do fatiamento.

Art. 2o Segundo artigo, tambem com texto longo o bastante para formar um
trecho independente e citavel de forma isolada.

Paragrafo unico. Disposicao acessoria do segundo artigo, com tamanho suficiente
para formar seu proprio trecho no indice.
"""


def test_separa_por_artigo():
    labels = [label for label, _ in split_articles(NORMA)]
    assert labels == [None, "Art. 1o", "Art. 2o", "Paragrafo unico"]


def test_texto_sem_artigo_vira_um_bloco():
    parts = split_articles("Texto corrido sem marcador de artigo.")
    assert len(parts) == 1
    assert parts[0][0] is None


def test_texto_vazio_nao_gera_bloco():
    assert split_articles("   \n  ") == []


def test_chunk_carrega_a_citacao():
    chunks = chunk_document("res-001", "Resolucao 001/2025", NORMA)
    art1 = next(c for c in chunks if c.citation.article == "Art. 1o")
    assert art1.citation.document_id == "res-001"
    assert str(art1.citation) == "Resolucao 001/2025, Art. 1o"


def test_ids_sao_unicos_e_ordenados():
    chunks = chunk_document("res-001", "Resolucao 001/2025", NORMA)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_artigo_longo_e_quebrado_em_janelas_com_sobreposicao():
    longo = "Art. 1o " + ("palavra " * 500)
    chunks = chunk_document("d", "D", longo, ChunkConfig(max_chars=400, overlap_chars=80))
    assert len(chunks) > 1
    assert all(c.citation.article == "Art. 1o" for c in chunks)


def test_trecho_curto_e_anexado_ao_seguinte():
    texto = "Art. 1o Curto.\n\nArt. 2o " + ("texto suficientemente longo " * 6)
    chunks = chunk_document("d", "D", texto, ChunkConfig(min_chars=100))
    assert len(chunks) == 1
    assert "Curto" in chunks[0].text


def test_config_invalida_e_rejeitada():
    with pytest.raises(ValueError, match="max_chars"):
        ChunkConfig(max_chars=0)
    with pytest.raises(ValueError, match="overlap_chars"):
        ChunkConfig(max_chars=100, overlap_chars=100)


def test_reconhece_artigo_com_simbolo_de_paragrafo():
    labels = [label for label, _ in split_articles("Art. 1o Texto.\n\n§ 1o Outro texto.")]
    assert "§ 1o" in labels
