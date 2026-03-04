#!/bin/bash
# 1. Wait for the running download (code_java + code_javascript, PID 471634) to finish
# 2. Download cc_news (replaces failed redpajama_books)
# 3. Append-tokenize all 4 new sources into existing train.bin + val.bin

set -euo pipefail

DOWNLOAD_PID=471634
PYTHON=/home/shashikant/anaconda3/envs/deepfill/bin/python
REPO=/mnt/zone/B/GPT/novamind-3b
LOG=/mnt/zone/A/datasets/append_tokenize.log
SOURCES="c4_en,cc_news,code_java,code_javascript"

echo "$(date): Waiting for download PID $DOWNLOAD_PID (code_java + code_javascript)..."
while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
    sleep 30
done
echo "$(date): code_java + code_javascript download done."

# cc_news wasn't in the original run — download it now
CC_NEWS_DIR=/mnt/zone/A/datasets/pretrain/cc_news
if [ ! -d "$CC_NEWS_DIR" ]; then
    echo "$(date): Downloading cc_news..."
    "$PYTHON" "$REPO/data/download.py" --stage pretrain \
        2>&1 | tee /mnt/zone/A/datasets/download_ccnews.log
    echo "$(date): cc_news download done."
else
    echo "$(date): cc_news already present, skipping download."
fi

echo "$(date): Starting incremental tokenization: $SOURCES"
cd "$REPO"
"$PYTHON" data/dataset.py \
    --stage append \
    --sources "$SOURCES" \
    2>&1 | tee "$LOG"

echo "$(date): All done. See $LOG"
