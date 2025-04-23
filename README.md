# bridge-ml-api

**bridge-ml-api** powers semantic tagging, embedding, and analysis of audio uploads for BRIDGE, an AI-based music collaboration platform. This backend leverages CLAP and TTMR++ embeddings, stem extraction, and LLM-driven caption/tag generation to provide real-time music similarity search and metadata enrichment.

---

## ✦ Key Features

- **CLAP + TTMR++ Dual Embedding**: Extracts high-dimensional semantic representations of audio.
- **Multi-vector FAISS Indexing**: Separates user-matching and semantic-tagging workflows for efficient vector search.
- **LLM-enhanced Metadata Generation**: Uses nearest-neighbor context to produce natural language tags and summaries.
- **Stem Extraction (Demucs)**: Optionally extracts and filters stems for instrument-aware analysis.
- **FastAPI Backend**: Powers real-time inference, search, and audio processing.
- **Lightweight Deployment**: All indices and metadata are under 100MB and versioned for seamless redeployment.

---

## 📁 Project Structure

```bash
bridge-ml-api/
├── services/
│   ├── clap_wrapper.py             # CLAP embedding + FAISS handling
│   ├── ttmrpp_wrapper.py           # TTMR++ embedding + FAISS handling
│   ├── audio_multi_processor.py    # Hybrid embedding pipeline (CLAP + TTMR++)
│   ├── audio_stemmer.py            # Demucs-based stem splitting and filtering
│   ├── taggen_llm.py               # GPT-style tag/summary generation
├── routes/
│   ├── semantic.py                 # Inference and upload API endpoints
├── configs/
│   ├── index_configs.py            # Paths to FAISS indices and metadata
├── scripts/
│   ├── build_fma_index.py          # Embeds FMA dataset using CLAP
│   ├── build_ttmr_index.py         # Embeds tracks with TTMR++
│   ├── download_clap_checkpoint.py # Downloads pretrained CLAP checkpoint
│   ├── download_ttmr_models.py     # Downloads TTMR++ model weights
│   └── download_fma_dataset.py     # Downloads the FMA dataset (small/medium)
├── .railway/
│   └── init.sh                     # Auto-setup script for submodules and models
```

## 🔧 Setup Instructions

### 1. Clone and activate environment

```bash
git clone https://github.com/HarryAhnHS/bridge-ml-api.git
cd bridge-ml-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Clone TTMR++ dependency

```bash
mkdir -p external
cd external
git clone https://github.com/seungheondoh/music-text-representation-pp.git
```

### 3. Download models

```bash
python scripts/download_clap_checkpoint.py
python scripts/download_ttmr_models.py
```

This will fetch:

- `music_speech_audioset_epoch_15_esc_89.98.pt` (CLAP)
- `best.pth` and configs for TTMR++

### 4. (Optional) Download FMA dataset

The following uses the FMA small subset (~8k tracks):

```bash
python scripts/download_fma_dataset.py
```

Source: https://github.com/mdeff/fma

---

## 🧠 Embedding Pipelines

### A. Build CLAP Index (FMA Tagging Index)

```bash
python scripts/build_fma_index.py
```

This script embeds all `.mp3` files in `fma_small/`, saves a FAISS index and a JSON metadata map.

### B. Build TTMR++ Index

```bash
python scripts/build_ttmr_index.py
```

Uses pretrained TTMR++ model to extract embeddings and save FAISS/metadata pair.

TTMR++ combines textual and audio knowledge with metadata association.  
Reference: https://arxiv.org/abs/2211.06687  
Official repo: https://github.com/seungheondoh/music-text-representation-pp

---

## 🚀 Deployment Notes

All required model files, FAISS indices, and metadata are <100MB and version-controlled (not LFS). For Railway or similar PaaS:

- Add `.railway/init.sh` to automatically clone TTMR++ and download checkpoints.
- Set environment variable `RAILWAY_DEPLOYMENT=true` to trigger auto-init on deploy.

---

## 🔍 Tech Stack

- Python, FastAPI
- FAISS (multi-index nearest neighbor search)
- CLAP (Contrastive Language-Audio Pretraining)
- TTMR++ (Text-to-Music Retrieval with metadata and LLMs)
- SentenceTransformers
- OpenAI / LLaMA2 (via Together API or local server)
- Demucs (stem separation)
- Next.js (used for paired frontend inference)

---

## 📚 References

- [CLAP: LAION Contrastive Language-Audio Pretraining](https://arxiv.org/abs/2211.06687)
- [TTMR++: Finetuned LLM + Metadata for Text-to-Music Retrieval](https://github.com/seungheondoh/music-text-representation-pp)
- [Free Music Archive Dataset (FMA)](https://github.com/mdeff/fma)

---

## License

MIT
