"""Geracao de resposta com citacao obrigatoria.

A regra do servico: **nenhuma resposta sem trecho que a sustente**. Quando a
recuperacao nao traz nada acima do limiar, o servico responde que nao encontrou
base normativa, em vez de deixar o modelo preencher a lacuna. Em contexto
administrativo, uma resposta inventada com aparencia de citacao e pior que
nenhuma resposta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from docsrag.types import Answer, ScoredChunk

NO_BASIS_MESSAGE = (
    "Nao encontrei base normativa nos documentos indexados para responder com seguranca. "
    "Reformule a pergunta ou verifique se a norma aplicavel foi ingerida."
)

SYSTEM_PROMPT = """Voce responde perguntas sobre normativos institucionais.

Regras que nao podem ser violadas:
1. Use exclusivamente os trechos fornecidos. Nao recorra a conhecimento proprio.
2. Cite a fonte de cada afirmacao no formato [id-do-trecho].
3. Se os trechos nao respondem a pergunta, diga isso explicitamente.
4. Nao interprete alem do texto. Aponte a ambiguidade quando ela existir.
5. Responda em portugues, de forma objetiva."""


class LLM(Protocol):
    """Contrato minimo de um modelo de linguagem."""

    def complete(self, system: str, user: str) -> str:
        """Devolve a resposta do modelo para o par de mensagens."""
        ...


class EchoLLM:
    """Modelo falso e deterministico, para teste e demonstracao.

    Devolve o primeiro trecho citado corretamente, o que permite exercitar todo
    o fluxo, incluindo a verificacao de sustentacao, sem chave de API.
    """

    def __init__(self, response: str | None = None) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        """Registra a chamada e devolve a resposta configurada."""
        self.calls.append((system, user))
        if self._response is not None:
            return self._response

        for line in user.splitlines():
            if line.startswith("[") and "]" in line:
                return f"Conforme o trecho {line[: line.index(']') + 1]}, ver o texto citado."
        return NO_BASIS_MESSAGE


def build_prompt(question: str, sources: list[ScoredChunk]) -> str:
    """Monta a mensagem do usuario com os trechos numerados."""
    blocks = [f"[{s.chunk_id}] ({s.chunk.citation})\n{s.chunk.text}" for s in sources]
    return "TRECHOS:\n\n" + "\n\n".join(blocks) + f"\n\nPERGUNTA: {question}"


def extract_cited_ids(text: str, valid_ids: set[str]) -> set[str]:
    """Extrai do texto os identificadores citados que existem entre os validos."""
    return {cid for cid in valid_ids if f"[{cid}]" in text}


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Parametros da geracao."""

    min_score: float = 0.0
    """Pontuacao minima de recuperacao para um trecho valer como fonte."""

    require_citation: bool = True
    """Se verdadeiro, resposta sem citacao valida e marcada como nao sustentada."""


class Generator:
    """Gera respostas sustentadas pelos trechos recuperados."""

    def __init__(self, llm: LLM, config: GeneratorConfig | None = None) -> None:
        self._llm = llm
        self._config = config or GeneratorConfig()

    def answer(self, question: str, sources: list[ScoredChunk]) -> Answer:
        """Responde a pergunta a partir dos trechos, ou recusa de forma explicita."""
        if not question.strip():
            raise ValueError("pergunta vazia")

        usable = [s for s in sources if s.score > self._config.min_score]

        if not usable:
            return Answer(text=NO_BASIS_MESSAGE, sources=[], grounded=False)

        raw = self._llm.complete(SYSTEM_PROMPT, build_prompt(question, usable))

        cited = extract_cited_ids(raw, {s.chunk_id for s in usable})

        if self._config.require_citation and not cited:
            return Answer(text=NO_BASIS_MESSAGE, sources=[], grounded=False)

        return Answer(
            text=raw,
            sources=[s for s in usable if s.chunk_id in cited] or usable,
            grounded=True,
        )
