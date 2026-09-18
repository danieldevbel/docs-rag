"""Avaliacao da recuperacao.

Sem metrica versionada, nao ha como saber se uma mudanca no fatiamento melhorou
ou piorou o sistema. O conjunto dourado fica em `eval/golden.json`, no
repositorio, e roda na CI junto com os testes: qualquer queda de recall aparece
no pull request, nao em producao.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from docsrag.retrieval import Retriever


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """Uma pergunta com os trechos que deveriam ser recuperados."""

    question: str
    relevant_ids: list[str]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("pergunta vazia no conjunto dourado")
        if not self.relevant_ids:
            raise ValueError(f"caso sem trecho relevante: {self.question}")


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Resultado agregado de uma avaliacao."""

    cases: int
    recall_at_k: float
    mrr: float
    k: int

    def meets(self, min_recall: float, min_mrr: float) -> bool:
        """Indica se o resultado atinge os limiares exigidos."""
        return self.recall_at_k >= min_recall and self.mrr >= min_mrr

    def as_dict(self) -> dict[str, float | int]:
        """Serializa o relatorio para registro em artefato de CI."""
        return {
            "cases": self.cases,
            "k": self.k,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
        }


def load_golden(path: Path) -> list[GoldenCase]:
    """Carrega o conjunto dourado de um arquivo JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenCase(question=c["question"], relevant_ids=c["relevant_ids"]) for c in raw]


def evaluate(retriever: Retriever, cases: list[GoldenCase], k: int = 5) -> EvalReport:
    """Mede recall@k e MRR do recuperador sobre o conjunto dourado.

    recall@k: fracao de casos em que ao menos um trecho relevante apareceu
    entre os k primeiros. MRR: media do inverso da posicao do primeiro trecho
    relevante, que penaliza o acerto que veio em quinto lugar.
    """
    if not cases:
        raise ValueError("conjunto dourado vazio")
    if k <= 0:
        raise ValueError("k deve ser positivo")

    hits = 0
    reciprocal_sum = 0.0

    for case in cases:
        retrieved = [s.chunk_id for s in retriever.search(case.question, k=k)]
        relevant = set(case.relevant_ids)

        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                hits += 1
                reciprocal_sum += 1.0 / rank
                break

    n = len(cases)
    return EvalReport(cases=n, recall_at_k=hits / n, mrr=reciprocal_sum / n, k=k)
