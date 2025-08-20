from fastapi import FastAPI
from routes.semantic import router as semantic_router
# from routes.instruments import router as instruments_router
import faiss
from configs.index_configs import TAGGING_INDEX, TTMR_INDEX, TTMR_ARTIST_INDEX, TAGGING_META, TTMR_META, TTMR_ARTIST_META
import json
import os
import subprocess

app = FastAPI()

@app.on_event("startup")
def load_faiss_indices():
    def load_json(path):
        try:
            with open(path, "r") as f:
                content = f.read().strip()
                return json.loads(content) if content else []
        except Exception as e:
            print(f"[WARN] Failed to load metadata from {path}: {e}")
            return []

    def load_index(path):
        try:
            print(f"[FAISS] Loading index: {path}")
            return faiss.read_index(str(path))
        except Exception as e:
            print(f"[ERROR] Failed to load FAISS index at {path}: {e}")
            return None

    print("[FAISS INIT] Loading FAISS indices at startup...")

    app.state.faiss_variants = {
        "tagging_clap": {
            "index": load_index(TAGGING_INDEX),
            "metadata": load_json(TAGGING_META)
        },
        "tagging_ttmr": {
            "index": load_index(TTMR_INDEX),
            "metadata": load_json(TTMR_META)
        },
        "tagging_ttmr_artist": {
            "index": load_index(TTMR_ARTIST_INDEX),
            "metadata": load_json(TTMR_ARTIST_META)
        },
    }
    print("[FAISS INIT] All indices and metadata loaded successfully ✅")

# Model downloading moved to singleton files for lazy loading
# @app.on_event("startup")
# def prepare_models():
#     # Download TTMR++ model if not present
#     if not os.path.exists("models/ttmrpp/best.pth"):
#         from scripts.download_ttmr_models import download_ttmrpp
#         download_ttmrpp()
#
#     # Download CLAP model if not present
#     clap_ckpt_path = "checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt"
#     if not os.path.exists(clap_ckpt_path):
#         from scripts.download_clap_checkpoint import download_checkpoint
#         download_checkpoint()


# Register the route
app.include_router(semantic_router, prefix="/semantic")
# app.include_router(instruments_router)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    
    # Production configuration optimized for Railway
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Railway typically allocates 1-2 CPU cores
        loop="asyncio",
        access_log=False,  # Reduce I/O overhead
        reload=False,
        # Enable concurrent request handling
        limit_concurrency=None,  # No artificial concurrency limit
        limit_max_requests=None,  # No request limit
        timeout_keep_alive=5,  # Keep connections alive for 5 seconds
    )