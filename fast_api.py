import io
import os
import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chromadb import PersistentClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from peft import PeftModel

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage


# ----------------------------
# Config / Logging
# ----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cxr-api")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LORA_PATH = os.getenv("LORA_PATH", "./lora_findings/lora")
CHROMA_PATH = os.getenv("CHROMA_PATH", "vectorDB")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "cxr_findings")

# Retrieval/pipeline defaults (you can tune from env)
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))
DEDUP_SIM_THRESHOLD = float(os.getenv("DEDUP_SIM_THRESHOLD", "0.9"))
CLUSTER_DISTANCE_THRESHOLD = float(os.getenv("CLUSTER_DISTANCE_THRESHOLD", "0.25"))

# Groq / LLM defaults
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


SYSTEM_MESSAGE_TEMPLATE = """
You are a radiology assistant.

You are given a list of factual FINDINGS extracted from similar chest X-ray cases.
These findings are already relevant and non-contradictory.

TASK:
- Rewrite the findings into a single, concise FINDINGS section.
- Merge overlapping statements.
- Preserve all factual information.
- Do NOT add new abnormalities.
- Do NOT infer diagnoses.
- Do NOT mention prior studies unless explicitly stated.

STYLE:
- Objective radiology language
- Past tense
- No bullet points
- No speculation

FINDINGS:
{findings}
""".strip()


# ----------------------------
# Global resources (loaded once)
# ----------------------------
processor: Optional[MedCLIPProcessor] = None
model: Optional[torch.nn.Module] = None
collection = None
llm: Optional[ChatGroq] = None


# ----------------------------
# Core functions (adapted from your script)
# Your original functions are path-based: encode_image_with_medclip(image_path) etc. :contentReference[oaicite:1]{index=1}
# Here we adapt them to accept PIL.Image so FastAPI uploads work.
# ----------------------------
def _ensure_loaded() -> None:
    if processor is None or model is None or collection is None or llm is None:
        raise RuntimeError("Server resources not initialized. Startup may have failed.")


def encode_single_text_with_medclip(text: str) -> np.ndarray:
    _ensure_loaded()
    model.eval()
    with torch.no_grad():
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        text_embeds = model.encode_text(input_ids=input_ids, attention_mask=attention_mask)
        emb = text_embeds[0]
        if emb.dim() > 1:
            emb = emb[0]

        emb = emb.detach().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


def encode_image_with_medclip_pil(image: Image.Image) -> np.ndarray:
    _ensure_loaded()
    model.eval()
    with torch.no_grad():
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        image_embeds = model.encode_image(pixel_values)
        emb = image_embeds[0]
        if emb.dim() > 1:
            emb = emb[0]

        emb = emb.detach().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


def retrieve_findings_for_image_emb(
    image_emb: np.ndarray,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    _ensure_loaded()
    results = collection.query(
        query_embeddings=[image_emb.tolist()],
        n_results=top_k,
        where={"split": "train"},  # same safety guard you used :contentReference[oaicite:2]{index=2}
    )

    # Robust empty handling (your original code assumes [0] exists) :contentReference[oaicite:3]{index=3}
    docs = (results.get("documents") or [[]])
    dists = (results.get("distances") or [[]])
    metas = (results.get("metadatas") or [[]])

    if not docs or not docs[0]:
        return []

    retrieved = []
    for i in range(len(docs[0])):
        retrieved.append(
            {
                "finding": docs[0][i],
                "distance": dists[0][i] if dists and dists[0] else None,
                "image_id": (metas[0][i] or {}).get("image_id") if metas and metas[0] else None,
            }
        )
    return retrieved


def filter_by_distance(retrieved_findings: List[Dict[str, Any]], max_distance: float = MAX_DISTANCE) -> List[Dict[str, Any]]:
    filtered = []
    for f in retrieved_findings:
        if f.get("distance") is None:
            continue
        if f["distance"] <= max_distance:
            filtered.append(f)
    return filtered


def deduplicate_findings(findings: List[Dict[str, Any]], similarity_threshold: float = DEDUP_SIM_THRESHOLD) -> List[str]:
    texts = [f["finding"] for f in findings]
    if not texts:
        return []

    embeddings = [encode_single_text_with_medclip(t) for t in texts]

    kept: List[str] = []
    kept_embeddings: List[np.ndarray] = []

    for text, emb in zip(texts, embeddings):
        is_duplicate = False
        for kept_emb in kept_embeddings:
            sim = cosine_similarity(emb.reshape(1, -1), kept_emb.reshape(1, -1))[0][0]
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(text)
            kept_embeddings.append(emb)

    return kept


def cluster_findings(findings: List[str], distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD) -> Dict[int, List[str]]:
    if not findings:
        return {}

    embeddings = [encode_single_text_with_medclip(f) for f in findings]

    # Note: your original used metric="cosine" :contentReference[oaicite:4]{index=4}
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    clusters: Dict[int, List[str]] = {}
    for label, finding in zip(labels, findings):
        clusters.setdefault(int(label), []).append(finding)
    return clusters


def produce_compact_findings(clusters: Dict[int, List[str]]) -> List[str]:
    compact: List[str] = []
    for label, items in clusters.items():
        if len(items) == 1:
            compact.append(items[0])
            continue

        embeddings = [encode_single_text_with_medclip(t) for t in items]
        sims = cosine_similarity(embeddings)
        avg_sims = sims.mean(axis=1)
        best_idx = int(avg_sims.argmax())
        compact.append(items[best_idx])

    return compact


def format_compact_findings(findings: List[str]) -> str:
    lines = []
    for i, text in enumerate(findings, start=1):
        lines.append(f"{i}. {text.strip()}")
    return "\n".join(lines).strip()


def image_text_alignment_score_from_emb(image_emb: np.ndarray, generated_text: str) -> Tuple[float, float]:
    text_emb = encode_single_text_with_medclip(generated_text)
    cosine_score = float(cosine_similarity(image_emb.reshape(1, -1), text_emb.reshape(1, -1))[0][0])
    percentage_score = cosine_score * 100.0
    return cosine_score, percentage_score


def run_full_pipeline(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Runs the whole pipeline and returns JSON-serializable output:
    - report (string)
    - metrics (dict)
    - compact_findings (list)
    - ranked_findings (string)
    """
    _ensure_loaded()

    # 1) Image embedding once
    image_emb = encode_image_with_medclip_pil(pil_image)

    # 2) Retrieval
    retrieved = retrieve_findings_for_image_emb(image_emb=image_emb, top_k=TOP_K)

    # If retrieval returns nothing, fail gracefully (no index errors like the original script bottom section) :contentReference[oaicite:5]{index=5}
    if not retrieved:
        return {
            "report": "",
            "ranked_findings": "",
            "compact_findings": [],
            "metrics": {
                "top_k": TOP_K,
                "retrieved_count": 0,
                "filtered_count": 0,
                "deduped_count": 0,
                "clusters_count": 0,
                "compact_count": 0,
                "alignment_cosine": None,
                "alignment_percent": None,
                "note": "No retrieval results from vector DB collection.",
            },
        }

    # 3) Distance filter
    filtered = filter_by_distance(retrieved, max_distance=MAX_DISTANCE)

    # 4) Deduplicate
    deduped = deduplicate_findings(filtered, similarity_threshold=DEDUP_SIM_THRESHOLD)

    # 5) Cluster
    clusters = cluster_findings(deduped, distance_threshold=CLUSTER_DISTANCE_THRESHOLD)

    # 6) Compact
    compact_findings = produce_compact_findings(clusters)

    # 7) Format findings list for prompt (same logic you had) :contentReference[oaicite:6]{index=6}
    ranked_findings = format_compact_findings(compact_findings)

    # 8) LLM
    prompt = SYSTEM_MESSAGE_TEMPLATE.format(findings=ranked_findings)
    report_msg = llm.invoke([SystemMessage(content=prompt)])
    report_text = (report_msg.content or "").strip()

    # 9) Alignment metrics
    if report_text:
        cos_sim, percent = image_text_alignment_score_from_emb(image_emb=image_emb, generated_text=report_text)
    else:
        cos_sim, percent = None, None

    metrics = {
        "top_k": TOP_K,
        "max_distance": MAX_DISTANCE,
        "dedup_similarity_threshold": DEDUP_SIM_THRESHOLD,
        "cluster_distance_threshold": CLUSTER_DISTANCE_THRESHOLD,
        "retrieved_count": len(retrieved),
        "filtered_count": len(filtered),
        "deduped_count": len(deduped),
        "clusters_count": len(clusters),
        "compact_count": len(compact_findings),
        "alignment_cosine": cos_sim,
        "alignment_percent": percent,
    }

    return {
        "report": report_text,
        "ranked_findings": ranked_findings,
        "compact_findings": compact_findings,
        "metrics": metrics,
    }


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="CXR Findings Report API", version="1.0")

# Allow React dev server by default (edit as you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_load_resources() -> None:
    global processor, model, collection, llm

    logger.info("Loading MedCLIP processor/model...")
    processor = MedCLIPProcessor()

    base_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    base_model.from_pretrained()

    if not os.path.isdir(LORA_PATH):
        raise RuntimeError(f"LoRA path not found: {LORA_PATH}")

    model_loaded = PeftModel.from_pretrained(base_model, LORA_PATH)
    model_loaded.to(DEVICE)
    model_loaded.eval()
    model = model_loaded

    logger.info("Connecting to ChromaDB...")
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Initializing Groq LLM client...")
    # ChatGroq reads API key from env (via load_dotenv above).
    llm = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE)

    logger.info("Startup complete.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    React should send multipart/form-data with key "file" (or change the param name).
    Returns:
      {
        "report": "...",
        "ranked_findings": "...",
        "compact_findings": [...],
        "metrics": {...}
      }
    """
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        # Torch/LLM work is blocking; for demo it’s fine. If needed, you can move to thread pool.
        result = run_full_pipeline(pil)
        return result
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
