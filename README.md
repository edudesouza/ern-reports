## *This repository presents the core components related to our paper.*

## Normalized GDPR JSON

JSON representation developed by the ADAPT Centre at Trinity College Dublin.

- [`gdpr.json`](gdpr.json)

## Ontology

A rule-based ontology focused on defining the deontic expressions—permission, prohibition, and obligation—contained in a text.

- [`_owl_gdpr.ttl`](./_owl_gdpr.ttl)

## Knowledge Graph

A rule-based knowledge graph in which each article is stored in full as an instance of the `Chunk` class. From each chunk, an LLM extracts the related ontology classes and creates a well-defined deontic rule, which is instantiated as an instance of the `Rule` class.

- [`kg.ttl`](kg.ttl)

| RDF Triples | Subjects | Predicates | Rule Classes |
| -----------: | -------: | ---------: | -----------: |
|        8,249 |    1,583 |         25 |          277 |

## Vector Store

The GDPR was vectorized using the same structure as the knowledge graph, embedding each full article to preserve its deontic coherence.

| Vector Size | Distance | HNSW | Data Type | Multivector |
| ----------: | -------- | ---- | --------- | ----------- |
| 384 | Cosine | Default | Dense: float embeddings (384 dimensions); sparse: BM25 indices and weights | Yes (hybrid dense and sparse) |

## Scripts

1. Class extraction and question expansion — Figure 1, A1

   - [`extraction.py`](./extraction.py)

2. SPARQL query and data retrieval — Figure 1, B1 and B2

   - [`graph_query.py`](./graph_query.py)

3. Reasoning and answer generation — Figure 1, C1 and D1

   - [`generation.py`](./generation.py)

4. Vector index construction

   - [`build_vector_index.py`](./build_vector_index.py)

5. Vector query and benchmark execution

   - [`run_benchmark_vector_rag.py`](./run_benchmark_vector_rag.py)

## Graph Experiments

| Parameter | Value |
| --- | --- |
| Query expansion | `gemini-2.5-flash temp: 0` |
| Models | Maritaca (`sabia-4`), Google (`gemini-2.5-flash-lite`), Mistral (`ministral`) |
| Evaluator (DeepEval) | `gpt-oss:120b-cloud temp: 0` |
| Metrics | SAF, SIM, NLI, AnswerRelevancy, Faithfulness |
| `threshold` | 0.70 |
| `max_chars` | 1500 |
| `top_k` | 50 |

## Vector Experiments

| Parameter | Value |
| --- | --- |
| Retriever | Hybrid (dense + sparse) |
| Dense model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Sparse model | `Qdrant/bm25` |
| Fusion | RRF (Reciprocal Rank Fusion) |
| Query expansion | `gemini-2.5-flash temp: 0` |
| Models | Maritaca (`sabia-4`), Google (`gemini-2.5-flash-lite`), Mistral (`ministral`) |
| Evaluator (DeepEval) | `gpt-oss:120b-cloud temp: 0` |
| Metrics | SAF, SIM, NLI, AnswerRelevancy, Faithfulness |
| `threshold` | 0.70 |
| `max_chars` | 1500 |
| `top_k` | 20 |

## Result Datasets

- [`gdpr_kaggle_1805a_gemini_harness.csv`](gdpr_kaggle_1805a_gemini_harness.csv)
- [`gdpr_kaggle_1805a_maritaca_harness.csv`](gdpr_kaggle_1805a_maritaca_harness.csv)
- [`gdpr_kaggle_1805a_ministral_harness.csv`](gdpr_kaggle_1805a_ministral_harness.csv)
- [`gdpr_kaggle_1805a_v_gemini.csv`](gdpr_kaggle_1805a_v_gemini.csv)

## Kaggle Q&A Dataset

- [`GDPR_qa_test_dataset_v2.csv`](_GDPR_qa_test_dataset_v2.csv)
