"""Testes da geracao com citacao obrigatoria."""

from __future__ import annotations

import pytest

from docsrag.generation import (
    NO_BASIS_MESSAGE,
    EchoLLM,
    Generator,
    GeneratorConfig,
    build_prompt,
    extract_cited_ids,
)
from docsrag.types import Chunk, Citation, ScoredChunk


def scored(cid: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=cid, text=f"Texto de {cid}.", citation=Citation("d", "Doc", "Art. 1o")
        ),
        score=score,
    )


def test_sem_fonte_recusa_responder():
    answer = Generator(EchoLLM()).answer("pergunta?", [])
    assert answer.grounded is False
    assert answer.text == NO_BASIS_MESSAGE


def test_fonte_abaixo_do_limiar_e_descartada():
    generator = Generator(EchoLLM(), GeneratorConfig(min_score=0.5))
    answer = generator.answer("pergunta?", [scored("c1", score=0.1)])
    assert answer.grounded is False


def test_resposta_sem_citacao_e_marcada_como_nao_sustentada():
    generator = Generator(EchoLLM("Resposta confiante sem citar nada."))
    answer = generator.answer("pergunta?", [scored("c1")])
    assert answer.grounded is False
    assert answer.text == NO_BASIS_MESSAGE


def test_citacao_inventada_nao_conta():
    generator = Generator(EchoLLM("Conforme [c99], sim."))
    answer = generator.answer("pergunta?", [scored("c1")])
    assert answer.grounded is False


def test_resposta_com_citacao_valida_e_aceita():
    generator = Generator(EchoLLM("Conforme [c1], sim."))
    answer = generator.answer("pergunta?", [scored("c1"), scored("c2")])
    assert answer.grounded is True
    assert [s.chunk_id for s in answer.sources] == ["c1"]


def test_citacao_pode_ser_desligada():
    generator = Generator(EchoLLM("Sem citar."), GeneratorConfig(require_citation=False))
    assert generator.answer("pergunta?", [scored("c1")]).grounded is True


def test_pergunta_vazia_e_rejeitada():
    with pytest.raises(ValueError, match="pergunta vazia"):
        Generator(EchoLLM()).answer("   ", [scored("c1")])


def test_prompt_inclui_id_e_citacao():
    prompt = build_prompt("qual o prazo?", [scored("c1")])
    assert "[c1]" in prompt
    assert "Doc, Art. 1o" in prompt
    assert "qual o prazo?" in prompt


def test_extract_ids_so_aceita_os_validos():
    assert extract_cited_ids("ver [a] e [z]", {"a", "b"}) == {"a"}


def test_citacoes_sao_unicas_e_ordenadas():
    generator = Generator(EchoLLM("Ver [c1] e [c2]."))
    answer = generator.answer("pergunta?", [scored("c1"), scored("c2")])
    assert len(answer.citations) == 1  # mesmo documento e artigo
