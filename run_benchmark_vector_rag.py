import asyncio
import csv

import os
import re
import time

# Imports do DeepEval
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from dotenv import load_dotenv
from rich import print

# Imports de Embeddings e Qdrant
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Imports do seu projeto
from src_en.config import settings
from src_en.evaluation import nli, saf, sim
from src_en.output import csv_create
from src_en.services import keywords_create, response_create
from src_en.utils import diff_time

load_dotenv()

os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "200"

# Configurações do Qdrant e dos modelos de embedding
COLLECTION_NAME = "gdpr_hybrid"
DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def build_retrieval_context(items, top_k=100, max_chars=1500, prefix_ids=True):
    if items is None:
        return []

    if isinstance(items, str):
        txt = normalize_ws(items)
        return (
            [txt[:max_chars] + ("..." if len(txt) > max_chars else "")] if txt else []
        )

    if isinstance(items, dict):
        items = [
            {"id_chunk": k, "texto_chunk": v, "score": 0} for k, v in items.items()
        ]

    items_sorted = sorted(items, key=lambda x: x.get("score", 0), reverse=True)
    seen = set()
    out = []

    for it in items_sorted:
        raw = (
            it.get("texto_chunk")
            or it.get("descricao_regra")
            or it.get("texto_regra")
            or ""
        )
        txt = normalize_ws(raw)
        if not txt:
            continue

        if prefix_ids:
            ident = it.get("id_chunk") or it.get("entidade") or ""
            if ident:
                txt = f"[{ident}] {txt}"

        if len(txt) > max_chars:
            txt = txt[:max_chars] + "..."

        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append(txt)
        if len(out) >= top_k:
            break

    return out


def perform_hybrid_search(query_text, client, dense_model, sparse_model, limit=5):
    """
    Realiza a busca híbrida (Dense + Sparse) usando Reciprocal Rank Fusion (RRF) no Qdrant local
    """
    # 1. Gerar vetores da pergunta
    dense_vec = dense_model.encode(query_text, normalize_embeddings=True).tolist()

    sparse_result = list(sparse_model.embed([query_text]))[0]
    sparse_vec = models.SparseVector(
        indices=[int(i) for i in sparse_result.indices],
        values=[float(v) for v in sparse_result.values],
    )

    # 2. Fazer a busca híbrida no Qdrant combinando os dois scores
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=limit),
            models.Prefetch(query=sparse_vec, using="sparse", limit=limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )

    # 3. Formatar saída para compatibilidade com o pipeline original
    dataset = []
    texts = []
    for point in search_result.points:
        payload = point.payload
        chunk_text = payload.get("text", "")
        uri = payload.get("uri", str(point.id))

        dataset.append(
            {"id_chunk": uri, "texto_chunk": chunk_text, "score": point.score}
        )
        texts.append(f"[{uri}] {chunk_text}")

    return {"dataset": dataset, "response": "\n\n".join(texts)}


async def main(
    id,
    resposta_gt,
    pergunta,
    qdrant_client,
    dense_model,
    sparse_model,
    retrieval_size=5,
    output=None,
    threshold=0.60,
):
    print("[red] \n--- inicio ---")

    if not pergunta:
        print("Nenhuma pergunta encontrada!")
        return

    inicio = time.time()
    inicio_global = time.time()

    print(f"\nUSER: {pergunta}")
    print("-" * 100, "\n")

    # --------------------------------------------------------------------------
    # 1. QUERY EXPANSION
    # --------------------------------------------------------------------------
    expantion = keywords_create(
        pergunta, "gemini-2.5-flash", settings.GEMINI_API_KEY, ""
    )

    palavras_chave   = expantion["keywords"]
    complexity_score = expantion["complexity_score"]
    query_canonical  = expantion["query_expansion"]["canonical"]

    print(f"\n-> canonical fact: {query_canonical}")
    print(f"-> rewriting: {palavras_chave}")

    if complexity_score < 0.55:
        print(f"-> complexidade alta: {complexity_score:.2f}")
        retrieval_size = 20
    else:
        print(f"-> complexidade baixa: {complexity_score:.2f}")

    diff_time("\n-> #1 expandir query OK: ", inicio)

    # --------------------------------------------------------------------------
    # 2. RETRIEVER HÍBRIDO (QDRANT)
    # --------------------------------------------------------------------------
    inicio = time.time()

    # Usamos o query expandido/canônico para melhorar a busca vetorial
    recuperacao = perform_hybrid_search(
        query_text=query_canonical,
        client=qdrant_client,
        dense_model=dense_model,
        sparse_model=sparse_model,
        limit=retrieval_size,
    )

    contexto = recuperacao["response"]
    knowledge = recuperacao["dataset"]

    diff_time("\n-> #2 buscar dados Qdrant, OK: ", inicio)

    # --------------------------------------------------------------------------
    # 3. LLM GENERATION
    # --------------------------------------------------------------------------
    inicio = time.time()

    response_llm = await asyncio.to_thread(
        response_create, palavras_chave, pergunta, contexto, "ollama"
    )
    resposta = response_llm["answer"]
    grounding = response_llm["chunks"]

    print(f"\nQuestion:    {pergunta}")
    print(f"\nLLM:         {resposta}")
    print(f"Grounding:     {grounding}")
    print(f"\nGold answer: {resposta_gt}")

    diff_time("\n-> #3 ground truth e resposta OK: ", inicio)

    # --------------------------------------------------------------------------
    # 4. METRICS (SAF, SIM, NLI)
    # --------------------------------------------------------------------------
    inicio = time.time()

    saf_score = saf(knowledge, resposta, pergunta, False)
    task_sim  = asyncio.to_thread(sim, resposta_gt, resposta)
    task_nli  = asyncio.to_thread(nli, resposta_gt, resposta)

    response_saf, score_sim_result, score_nli_result = await asyncio.gather(
        saf_score, task_sim, task_nli
    )

    nli_val = score_nli_result["score"]
    sim_val = score_sim_result["score"]

    print(f"-> nli: {nli_val:.2f}")
    print(f"-> sim: {sim_val:.2f}")
    print(f"-> saf: {response_saf:.2f}")

    diff_time("\n-> #4 factualidade e comparação: ", inicio)

    # --------------------------------------------------------------------------
    # 5. DEEPEVAL
    # --------------------------------------------------------------------------
    avaliador = "gpt-oss:120b-cloud"  # Mantenha o seu modelo do Ollama
    print(f"\n-> avaliador deepeval: {avaliador}")

    model = OllamaModel(
        model=avaliador, base_url="http://localhost:11434", temperature=0
    )

    answer_relevancy = AnswerRelevancyMetric(model=model, include_reason=True)
    faithfulness = FaithfulnessMetric(model=model, include_reason=True)

    retrieval_ctx = build_retrieval_context(recuperacao["dataset"], top_k=20)

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta,
        expected_output=resposta_gt,
        retrieval_context=retrieval_ctx,
        context=[recuperacao["response"]],
    )

    try:
        answer_relevancy.measure(test_case)
        print("- Relevancia: ", answer_relevancy.score)
        print("- Reason: ", answer_relevancy.reason)
    except Exception as erro:
        print(f"ERRO relevancia: {erro}")
        answer_relevancy.score = 0

    print("-" * 50)

    try:
        faithfulness.measure(test_case)
        print("- Confiabilidade: ", faithfulness.score)
        print("- Reason: ", faithfulness.reason)
    except Exception as erro:
        print(f"ERRO confiabilidade: {erro}")
        faithfulness.score = 0

    print("-" * 100)

    # --------------------------------------------------------------------------
    # 6. OUTPUT & LÓGICA DE AVALIAÇÃO
    # --------------------------------------------------------------------------
    if output:
        csv_create(
            output,
            id,
            "vector_hybrid",
            pergunta,
            resposta_gt,
            response_llm,
            complexity_score,
            response_saf,
            nli_val,
            sim_val,
            answer_relevancy.score,
            faithfulness.score,
            "gemma",
        )

    if response_saf > threshold and (nli_val > threshold or sim_val > threshold):
        print("[green]*** Resposta APROVADA: Fiel ao documento e ao GT ***\n")
    elif response_saf > threshold and (nli_val <= threshold and sim_val <= threshold):
        print(
            "[yellow]*** REVISÃO NECESSÁRIA: IA fiel ao documento, mas diverge do GT ***\n"
        )
    elif response_saf <= threshold:
        print("[red]*** Resposta NEGADA: alto grau de ambiguidade ***\n")
    else:
        print(
            "[magenta]*** Resposta NEGADA: Inconsistência (Bate com GT, mas SAF baixo) ***\n"
        )

    diff_time("-> Tempo total da pergunta: ", inicio_global)
    print("[red] --- fim ---\n")


async def run_batch(
    json_path="_GDPR_qa_test_dataset_v2.csv",
    output_csv="gpdr_kaggle_vector_hybrid_deepeval.csv",
):
    print("\n--- Iniciando Setup de Modelos e Conexões ---\n")
    inicio_setup = time.time()

    # 1. Carrega os modelos na memória APENAS UMA VEZ
    print("Carregando modelo denso...")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)

    print("Carregando modelo esparso...")
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)

    print("Conectando ao banco Qdrant...")
    qdrant_client = QdrantClient(path="data/vector_indexes/qdrant")

    diff_time("Tempo de Setup: ", inicio_setup)
    print("\n--- Iniciando Benchmark ---\n")

    inicio_benchmark = time.time()

    with open(json_path, "r", encoding="utf-8") as f:
        perguntas = list(csv.reader(f))

    total = len(perguntas) - 1

    for index, item in enumerate(perguntas[1:], start=1):
        id_pergunta = str(index)
        pergunta = item[0]
        resposta_gt = item[1]

        await main(
            id=id_pergunta,
            resposta_gt=resposta_gt,
            pergunta=pergunta,
            qdrant_client=qdrant_client,
            dense_model=dense_model,
            sparse_model=sparse_model,
            retrieval_size=10,
            output=output_csv,
        )

        print(f"Progresso: {index} de {total}")

    qdrant_client.close()
    diff_time("\n-> Fim do Benchmark Híbrido: ", inicio_benchmark)


if __name__ == "__main__":
    asyncio.run(run_batch())
