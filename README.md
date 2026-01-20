# Chest X-Ray Report Generation using Fine-Tuned MedCLIP

This project implements a complete pipeline for **Chest X-Ray image analysis and report generation** using a **fine-tuned MedCLIP model**, a **vector database**, and a **FastAPI backend connected to a React frontend**.

The system retrieves relevant radiology findings based on an input X-ray image and generates a concise radiology-style report.

---

## Dataset

The dataset used in this project is:

**Hugging Face Dataset:**  
`itsanmolgupta/mimic-cxr-dataset`

- Downloaded from Hugging Face
- Contains chest X-ray images and corresponding radiology reports
- Used for fine-tuning and embedding storage
- The dataset is **not included** in this repository and must be downloaded locally

---

## Model and Encoding Strategy

- Base model: **MedCLIP**
- Fine-tuning method: **LoRA (Low-Rank Adaptation)**
- Fine-tuned weights are stored in: lora_findings/

  
---

### 2. Storing Embeddings in Vector Database
**Notebook:** `storing.ipynb`

- Loads the pretrained MedCLIP model + LoRA weights
- Encodes radiology findings text
- Stores embeddings into a vector database (ChromaDB)
- The vector database is stored locally in: vectorDB/

  
> Note: `vectorDB/` is ignored by Git and must be generated locally.

---

### 3. Inference and Retrieval Pipeline
**File:** `fast_api.py`

This file contains the **entire inference pipeline**, including:

1. Loading MedCLIP with LoRA fine-tuned weights
2. Encoding the input chest X-ray image
3. Retrieving top-K similar findings from the vector database
4. Filtering results using a distance threshold
5. Deduplicating semantically similar findings
6. Clustering findings to remove redundancy
7. Selecting representative findings
8. Generating a final radiology-style report
9. Computing alignment metrics between the image and generated report

---

### 4. FastAPI Backend

- Implemented in: `fast_api.py`
- Exposes a FastAPI endpoint: POST /predict


- Input: Chest X-ray image (`multipart/form-data`)
- Output:
```json
{
  "report": "...",
  "ranked_findings": "...",
  "compact_findings": [...],
  "metrics": {...}
}

This backend is used to connect the full pipeline to the frontend.

## Installation and Setup

### 1. Clone the Repository

git clone <repository-url>
cd tutore

2. Create and Activate Virtual Environment
Create the virtual environment:

python -m venv .tutore_venv
Activate it depending on your OS

3. Install Dependencies
pip install -r requirements.txt

4. Environment Variables
Create a .env file at the root of the project and add:
GROQ_API_KEY=your_api_key_here

5. Run the FastAPI Backend
uvicorn fast_api:app --reload --host 127.0.0.1 --port 8000
http://127.0.0.1:8000/docs

6. Run the Frontend
cd front/genreport
npm install
npm run dev

7.Project Structure
tutore/
│
├── fast_api.py              # Full FastAPI + inference pipeline
├── requirements.txt
├── .gitignore
├── .env
│
├── lora_findings/           # Fine-tuned LoRA weights
├── vectorDB/                # Vector database (generated locally)
│
├── CLIP_fine_tuning.ipynb   # MedCLIP fine-tuning notebook
├── storing.ipynb            # Vector DB creation notebook
│
└── front/                   # React frontend

8.Metrics
The system provides the following metrics:

- Number of retrieved findings

- Number of filtered and deduplicated findings

- Number of clusters

- Image–text cosine similarity

- Alignment percentage score

