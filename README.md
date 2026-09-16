# ConceptShelf — Multilingual Semantic Book Search Engine

A two-stage neural information retrieval system that lets users discover books by mood, theme, or vague plot descriptions without needing exact keyword matches.

Built with **PyTorch**, **SentenceTransformers**, **FastAPI**, and a retro-editorial vanilla web frontend.

book data from: https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/5

---

## Architecture Overview
```
[User Query]
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1: Dense Retrieval (Bi-Encoder)                    │
│ Model: intfloat/multilingual-e5-small                    │
│ Fast GPU cosine similarity on pre-computed embeddings    │
│ Reduces 20,000+ books down to 25 top candidates (<5ms)  │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2: Neural Reranking (Cross-Encoder)                │
│ Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1        │
│ Deep cross-attention between query & candidate blurbs    │
│ Outputs calibrated relevance probabilities (~30ms)       │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3: Score Calibration & Metadata Weighting          │
│ Blends normalized semantic probability with 5-star       │
│ rating distributions to boost high-quality matches       │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
                    [Top 5 Final Results]
```
---

## Empirical Benchmark & Evaluation

The retrieval pipeline is evaluated using a custom multilingual test suite (English, German, and Traditional Chinese) testing abstract plot descriptions against known ground-truth books.

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Hit Rate @ 10** | **82.35%** | Target book retrieved in top 10 results (14 of 17 hits) |
| **MRR @ 10** | **0.61** | Mean Reciprocal Rank (evaluates how close to Rank 1 the target appears) |
| **Query Latency** | **~350ms** | End-to-end inference on CUDA (Bi-Encoder + Cross-Encoder) |

### Sample Evaluations

* **Thematic / Semantic abstraction:**  
  Query: *"A young man stays youthful while his hidden portrait ages and decays"*  
  $\rightarrow$ **Rank 1 (99.7% conf):** *The Picture of Dorian Gray*

* **Cross-lingual thematic match (German $\rightarrow$ English catalog):**  
  Query: *"Zwei verfeindete Familien führen ihre jugendlichen Liebenden in den gemeinsamen Tod"*  
  $\rightarrow$ **Rank 2 (96.9% conf):** *Romeo and Juliet*

---

## Key Engineering Details

1. **Memory-Mapped Vectors (`np.memmap`):** Pre-computed embeddings are mapped directly to memory without reading the full array into RAM, allowing instant startup and low memory overhead.
2. **GPU Top-K Selection:** Replaced slow $O(N \log N)$ NumPy sorting on CPU with PyTorch’s native `torch.topk` on CUDA, accelerating candidate filtering to under 5ms.
3. **Data Quality & Hygiene:** Scraped catalogs were cleaned to remove HTML artifacts, deduplicate multi-edition book prints and filter out sparse metadata
4. **Multi-Screen UI:** Minimalist, editorial frontend with zero build-tooling dependencies (Vanilla HTML/CSS/JS).

---

## Quickstart

### 1. Prerequisites & Installation

```bash
git clone https://github.com/momixer1/booksearch.git
cd booksearch

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install sentence-transformers pandas numpy fastapi uvicorn
```
To install with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Run the Benchmark

```bash
python benchmark.py
```

### 3. Launch the Web App

```bash
uvicorn main:app --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Future Roadmap

- [ ] **Vector Indexing (HNSW / FAISS):** Replace dense linear scans with Approximate Nearest Neighbor (ANN) graphs to support 500,000+ titles.
- [ ] **Learning to Rank (LTR):** Replace the Stage 3 metadata heuristic with a LambdaMART model (via LightGBM) trained on click logs.
- [ ] **Embedding Quantization:** Compress vectors from `float32` to `int8` to reduce VRAM consumption by 75%.
