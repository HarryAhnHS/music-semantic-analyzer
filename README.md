# bridge-ml-api

cd /Users/harryahn/repos/bridge-ml-api
source .venv/bin/activate

## FMA Tagging Index status
💾 Saving FAISS index and metadata...

🎉 Done in 3213.9s
✔️  New tracks processed: 19806
↪️  Skipped existing (preloaded): 5179
🚫 Skipped missing files: 81574
⚠️  Skipped invalid embeddings: 0


## TTMR++ 
“Clone TTMR++ under external/music-text-representation-pp before running embedding scripts.”


# deployment pipeline
- all indices, meta json < 100MB on git
- .railway/init.sh script for submodule (should be auto)
- best.pth for ttmr (download_ttmr_models)
- music_speech_audioset_epoch_15_esc_89.98.pt for CLAP (download_clap_checkpoint script)