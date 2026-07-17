import json
import re
import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "gdpr_hybrid"
DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"


class VectorIngestion:
    
    def __init__(self, client, collection_name, dense_model, sparse_model):
        self.client = client
        self.collection_name = collection_name
        self.dense_model = dense_model
        self.sparse_model = sparse_model

    def setup_collection(self, dense_dim: int):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def ingest_json(self, json_path):
        
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        
        points = []
        for chapter in data["chapters"]:
            chapter_num = chapter["number"]
            for item in chapter["contents"]:
                articles = []
                if item["type"] == "article":
                    articles.append(item)
                elif item["type"] == "section":
                    articles.extend(
                        c for c in item["contents"] if c["type"] == "article"
                    )
                
                for article in articles:
                    article_num = article["number"]
                    for idx, point in enumerate(article["contents"], start=1):
                        text = clean_text(point.get("text", ""))
                        for sub in point.get("subpoints", []):
                            text += " " + clean_text(sub.get("text", ""))
                        
                        if not text:
                            continue
                        
                        dense_vec = self.dense_model.encode(
                            text, normalize_embeddings=True
                        ).tolist()
                        
                        sparse_result = list(self.sparse_model.embed([text]))[0]
                        sparse_vec = models.SparseVector(
                            indices=[int(i) for i in sparse_result.indices],
                            values=[float(v) for v in sparse_result.values],
                        )
                        
                        payload = {
                            "uri": f"Chapter_{chapter_num}/Article_{article_num}/Point_{idx}",
                            "text": text,
                            "chapter": chapter_num,
                            "article": article_num,
                        }

                        points.append(
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector={"dense": dense_vec, "sparse": sparse_vec},
                                payload=payload,
                            )
                        )

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"{len(points)} chunks inseridos no Qdrant.")

def clean_text(text: str):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def main():

    print("Iniciando o script...", flush=True)
    
    # =========================================================
    # LOAD MODEL
    # =========================================================
    print("Carregando modelo denso (SentenceTransformers)...", flush=True)
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    embedding_dim = dense_model.get_embedding_dimension()
    if embedding_dim is None:
        raise ValueError("Embedding dimension is None.")
    dense_dim = int(embedding_dim)

    print(f"Modelo denso carregado. Dimensão: {dense_dim}", flush=True)
    print("Carregando modelo esparso (FastEmbed)...", flush=True)

    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
    print("Modelo esparso carregado com sucesso.", flush=True)

    # =========================================================
    # QDRANT & INGESTION
    # =========================================================
    print("Conectando ao Qdrant...", flush=True)
    client = QdrantClient(path="data/vector_indexes/qdrant")
    print("Iniciando ingestão...", flush=True)
    
    ingestor = VectorIngestion(
        client=client, 
        collection_name=COLLECTION_NAME,
        dense_model=dense_model,
        sparse_model=sparse_model
    )

    ingestor.setup_collection(dense_dim=dense_dim)
    ingestor.ingest_json("data/raw/gdpr.json")

    print("Vetor de índices criado com sucesso.")
    client.close()

if __name__ == "__main__":
    main()