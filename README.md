# docs-rag

RAG sobre normativos institucionais. Recuperação híbrida, fatiamento que preserva o artigo, citação obrigatória e avaliação versionada que roda na CI. Se a norma não sustenta a resposta, o serviço diz que não sabe.

*Retrieval-augmented generation over institutional regulations: article-aware chunking, hybrid retrieval, mandatory citation, and a versioned evaluation set enforced in CI.*

![ci](https://github.com/danieldevbel/docs-rag/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Problema

Quem trabalha com normativo não quer um resumo plausível, quer o artigo. Um chatbot que responde "o limite para dispensa é de dez mil reais" sem dizer de onde tirou é inútil para instruir um processo, e perigoso se estiver errado. Três falhas típicas de RAG jurídico e administrativo:

1. **Fatiamento cego corta o artigo ao meio.** Uma janela de 512 tokens parte o caput do parágrafo, e a resposta cita metade da regra.
2. **Busca densa perde o termo exato.** O usuário procura "inexigibilidade"; o embedding traz "contratação direta", que é outra coisa.
3. **O modelo preenche a lacuna.** Sem trecho pertinente, ele responde assim mesmo, com aparência de citação.

Este projeto trata as três como requisito, não como ajuste de prompt.

## Resultado medido

Sobre o corpus de exemplo (2 resoluções fictícias, 21 trechos) e o conjunto dourado de 14 perguntas versionado em [`eval/golden.json`](eval/golden.json):

| Métrica | k=3 | k=5 |
|---|---|---|
| recall@k | 0,929 | **1,000** |
| MRR | 0,893 | **0,911** |

Reproduza com `make eval`. O limiar mínimo está em [`tests/test_evaluation.py`](tests/test_evaluation.py) e a CI falha se o recall cair abaixo de 0,85 ou o MRR abaixo de 0,70. Há também um teste que verifica que a métrica **reage**: um fatiamento propositalmente agressivo precisa produzir MRR pior, senão a métrica não está medindo nada.

## Como as três falhas são tratadas

**Fatiamento que preserva o artigo.** `chunking.py` quebra o texto nos marcadores `Art. Nº`, `Parágrafo único` e `§ Nº` antes de qualquer contagem de caracteres. Só quando o artigo excede o limite é que entra a janela deslizante com sobreposição, e mesmo aí todos os pedaços herdam a mesma citação. Trechos curtos demais são anexados ao seguinte em vez de virar ruído no índice.

**BM25 escrito à mão, não importado.** Em normativo a busca lexical costuma vencer a densa, porque o usuário procura o termo da lei. A implementação está em `retrieval.py`, com normalização de acento (`licitação` casa com `licitacao`) e remoção de palavras vazias. Está no repositório em vez de vir de uma biblioteca justamente para ser inspecionável: dá para abrir e ver por que um trecho ficou em primeiro.

Para combinar com busca densa, `reciprocal_rank_fusion` funde os rankings pela posição, não pela pontuação. Somar BM25 com cosseno diretamente não funciona, porque as escalas não são comparáveis.

**Citação obrigatória, verificada em código.** O gerador extrai do texto produzido os identificadores citados e confere contra os trechos realmente recuperados. Resposta sem citação, ou com citação de um identificador que não existe, é rejeitada e substituída por uma recusa explícita. Isso é um teste, não uma instrução de prompt:

```python
def test_citacao_inventada_nao_conta():
    generator = Generator(EchoLLM("Conforme [c99], sim."))
    answer = generator.answer("pergunta?", [scored("c1")])
    assert answer.grounded is False
```

## Arquitetura

```
documentos ──► fatiamento ──► índice BM25 ──┐
  (.txt)      (por artigo)                  ├──► fusão RRF ──► geração ──► verificação
                                índice denso┘                              de citação
                                (opcional)                                      │
                                                                    resposta ou recusa
```

```
src/docsrag/
  types.py        Citation, Chunk, Answer: citação é parte do tipo, não metadado solto
  chunking.py     quebra por artigo, janela deslizante como exceção
  corpus.py       ingestão do sistema de arquivos
  retrieval.py    BM25, busca densa e fusão por posição
  generation.py   prompt, protocolo LLM e verificação de sustentação
  evaluation.py   recall@k e MRR sobre o conjunto dourado
  api.py          FastAPI: /health, /search, /ask
  cli.py          linha de comando
```

Nenhum módulo do núcleo importa cliente de LLM. `LLM` é um `Protocol` e `EchoLLM` é a implementação determinística usada nos testes, o que permite exercitar **os 53 testes sem chave de API e sem rede**.

## Como rodar

```bash
git clone https://github.com/danieldevbel/docs-rag.git
cd docs-rag
make install
make check      # lint, tipos e testes
make eval       # métricas de recuperação
```

```bash
uv run docs-rag index
uv run docs-rag search "dispensa de licitacao"
uv run docs-rag ask "Qual o prazo de analise da requisicao pela area de suprimentos?"
uv run docs-rag eval --k 3 --strict
```

Saída de `ask`:

```
Conforme o trecho [res-001-2025#0008], ver o texto citado.

Fontes:
  - Resolucao Normativa 001/2025, Art. 7o
```

Com corpus próprio: coloque os `.txt` numa pasta, aponte `RAG_CORPUS_DIR` para ela e reescreva `eval/golden.json` com perguntas reais e os identificadores corretos. Sem refazer o conjunto dourado, as métricas não significam nada.

## Corpus de exemplo

Os dois documentos em [`corpus/`](corpus/) são **fictícios**. Imitam a linguagem de uma resolução normativa brasileira para exercitar o fatiamento por artigo, mas não reproduzem nenhuma norma real de nenhuma instituição.

## Configuração

Por variável de ambiente, prefixo `RAG_`. Ver [.env.example](.env.example).

| Variável | Padrão | Efeito |
|---|---|---|
| `RAG_CORPUS_DIR` | `corpus` | Pasta dos normativos |
| `RAG_CHUNK_MAX_CHARS` | `1200` | Tamanho antes de recorrer à janela deslizante |
| `RAG_CHUNK_OVERLAP_CHARS` | `150` | Sobreposição entre janelas |
| `RAG_TOP_K` | `5` | Trechos enviados ao modelo |
| `RAG_REQUIRE_CITATION` | `true` | Rejeitar resposta sem citação válida |
| `RAG_MIN_RECALL_AT_K` | `0.8` | Limiar de recall para `eval --strict` |
| `RAG_MIN_MRR` | `0.6` | Limiar de MRR para `eval --strict` |

## Decisões e alternativas descartadas

- **BM25 próprio em vez de `rank_bm25` ou Elasticsearch.** Abaixo de algumas dezenas de milhares de trechos, o custo é irrelevante e a legibilidade compensa: o repositório serve para mostrar que o autor entende o ranqueador, não que sabe instalá-lo. Acima disso, troque por Elasticsearch ou Qdrant sem tocar no restante, já que `Retriever` é um `Protocol`.
- **Fusão por posição (RRF) em vez de soma ponderada.** Pesos exigem calibração por corpus e envelhecem mal. RRF não exige escala comum e funciona bem sem ajuste.
- **Verificação de citação por conferência de identificador, não por segundo modelo.** Um verificador baseado em LLM custa outra chamada e erra também. Conferir se o identificador citado existe entre os recuperados é determinístico, instantâneo e pega o caso que mais importa, que é a citação inventada.
- **Conjunto dourado pequeno e versionado.** Catorze perguntas escritas à mão valem mais que mil geradas automaticamente, porque cada uma tem gabarito conferido no texto da norma.

## Limitações conhecidas

- **Verificação de citação não é verificação de fidelidade.** O serviço garante que a resposta cita um trecho recuperado, não que a afirmação corresponde ao que o trecho diz. Detectar isso exige avaliação de *groundedness* por juiz externo, que não está incluída.
- **Parágrafo único vira trecho isolado.** Ele frequentemente depende do caput para fazer sentido, e recuperado sozinho perde contexto. A correção é incluir o caput no trecho do parágrafo, que ainda não está implementado.
- **O corpus só lê `.txt`.** PDF com coluna dupla e OCR ruim, que é o formato real da maioria dos normativos, exige uma etapa de extração anterior.
- **A busca densa existe, mas sem extrator de embedding embutido.** `DenseRetriever` opera sobre vetores já calculados; conectar um modelo de embedding é escolha de quem usa, com implicação de custo e de envio de dados a terceiros.
- **Sem reranqueamento por cross-encoder.** É o próximo ganho de MRR, ao custo de latência.
- **`EchoLLM` não é um modelo.** Ele existe para tornar o fluxo testável. Com um modelo real, as métricas de recuperação continuam válidas, mas a qualidade da redação passa a depender dele.

## Licença

MIT. Ver [LICENSE](LICENSE).
