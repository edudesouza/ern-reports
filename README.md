## *This repository has the intention to present the core compoments related to our paper*

## Ontology
Rule based ontology, this ontologie is focus on define the deontic (permisson,prohibition,oblication) expression of a text
- [`_owl_gdpr.ttl`](./_owl_gdpr.ttl)

## Knowledge graph
Rule based KG, each article is full injected in a class called chunck, from wich a LLM capture its related classes, based on the ontology and creates a well defined deontic rule, wich is instaciated in a class name Rule
-[`kg.ttl`](kg.ttl)

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





## GDPR json normalized
gdpr.json

## Result datasets:
- gdpr_kaggle_1805a_gemini_harness.csv
- gdpr_kaggle_1805a_maritaca_harness.csv
- gdpr_kaggle_1805a_ministral_harness.csv
- gdpr_kaggle_1805a_v_gemini.csv

## kaggle Q&A dataset
_GDPR_qa_test_dataset_v2.csv
