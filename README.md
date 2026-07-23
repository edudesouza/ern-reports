## *This repository has the intention to present the core compoments related to our paper*

## GDPR json normalized
Json respresentation developed by The ADAPT Centre at Trinity College Dublin.
- [`gdpr.json`](gdpr.json)

## Ontology
Rule based ontology, this ontologie is focus on define the deontic (permisson,prohibition,oblication) expression of a text.
- [`_owl_gdpr.ttl`](./_owl_gdpr.ttl)

## Knowledge graph
Rule based KG, each article is full injected in a class called chunck, from wich a LLM capture its related classes, based on the ontology and creates a well defined deontic rule, wich is instanciated in a class name Rule.
- [`kg.ttl`](kg.ttl)
  
| RDF Triples | Subjects | Predicates | Rule Classes |
| ----------: | -------: | ---------: | -----------: |
|       8,249 |    1,583 |         25 |          277 |

## Vector store
The GDPR was vectorized using the same structure as the KG, embedding each full article to preserve its deontic coherence.
| Vector | Distance | HNSW | Data type | Multivector |
| -----: | -------- | ---- | --------- | ----------- |
| 384 | Cosine | Default | Dense: float embeddings (384 dims), Sparse: BM25 indices/weights | Yes (hybrid dense + sparse) |

## Scripts

1. Class extraction and question expantions, Figure 1, A 1
- [`extraction.py`](./extraction.py)

2. SPARQL query and data retrieval, Figure 1, B 1 and 2
- [`graph_query.py`](./graph_query.py)

3. Reasoning and answer generation, Figure 1, C 1 and D 1
- [`generation.py`](./generation.py)

4. Vector build
- [`build_vector_index.py`](./build_vector_index.py)

5. Vector query
- [`run_benchmark_vector_rag.py`](./run_benchmark_vector_rag.py)

## Graph Experiments


## Vector Experiments
| Parâmetro | Valor |
| --- | --- |
| Retriever | Hybrid (Dense + Sparse) |
| Dense model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Sparse model | `Qdrant/bm25` |
| Fusion | RRF (Reciprocal Rank Fusion) |
| Retrieval size | 10 (default) / 20 (complexidade alta) |
| Query expansion | `gemini-2.5-flash` |
| LLM model | `gemini 2.5, Ministral, Sabiá 4` |
| Evaluator (DeepEval) | `gpt-oss:120b-cloud` |
| `temperature` (avaliador) | 0 |
| Métricas | SAF, SIM, NLI, AnswerRelevancy, Faithfulness |
| `threshold` | 0.70 |
| `max_chars` | 1500 |
| `top_k` (contexto) | 20 |

## Result datasets:
- gdpr_kaggle_1805a_gemini_harness.csv
- gdpr_kaggle_1805a_maritaca_harness.csv
- gdpr_kaggle_1805a_ministral_harness.csv
- gdpr_kaggle_1805a_v_gemini.csv

## kaggle Q&A dataset
_GDPR_qa_test_dataset_v2.csv
