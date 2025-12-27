# SewingSiT

**SewingSiT** 是一個將 SiT (Scalable Interpolant Transformers) 改造為開放詞彙文本生成模型的專案。我們整合了 SSD-1B 的 VAE 與 CLIP Text Encoder，實現了從無到有的文生圖訓練管道。

> 📖 **關於技術細節、架構修改與設計理念，請參閱完整報告：[REPORT.md](REPORT.md)**

---

## 🚀 1. 環境安裝 (Installation)

請使用 Conda 建立虛擬環境。

**標準安裝 (Standard GPU)**:
```bash
conda env create -f environment.yml -n sewingSiT
conda activate sewingSiT
```

**新一代顯卡 (RTX 50 系列 / CUDA 12.4+)**:
```bash
conda env create -f environment_for50.yml -n sewingSiT_50
conda activate sewingSiT_50
```

---

## ⚡ 2. 快速開始 (Quick Start)

按照以下順序，您可以在 10 分鐘內跑通從資料下載到模型生成的完整流程。

### Step 1: 下載與預處理資料
我們使用 MS-COCO 驗證集 (約 1GB) 進行快速測試。腳本會自動下載圖片、標註，並將其轉換為訓練用的 Latents (`.npz`)。

```bash
# 1. 下載 COCO Val2017
python scripts/download_coco.py --target_dir /mnt/f/nccu/project/data/2dImg --split test_small

# 2. 預處理 (圖片尺寸 256x256 -> Latent 32x32)
python scripts/process_coco.py \
  --img_dir /mnt/f/nccu/project/data/2dImg/val2017 \
  --ann_file /mnt/f/nccu/project/data/2dImg/annotations/captions_val2017.json \
  --output_dir /mnt/f/nccu/project/data/2dImg/processed_val \
  --size 256
```

### Step 2: 訓練模型
使用 Rectified Flow (Transport) 進行訓練。

```bash
python scripts/train_transport.py \
  --data_dir /mnt/f/nccu/project/data/2dImg/processed_val \
  --epochs 10 \
  --batch_size 32 \
  --img_size 32 \
  --use_wandb \
  --run_name "quick-start-run"
```
*   `--img_size 32`: 對應 256x256 的原圖 (VAE 壓縮率為 8)。
*   `--use_wandb`: 強烈建議開啟以監控 Loss 曲線。

### Step 3: 生成圖片 (Inference)
訓練完成後，權重會儲存在 `checkpoints/`。使用以下指令生成圖片：

```bash
python scripts/sample.py \
  --checkpoint checkpoints/quick-start-run.pt \
  --prompt "A red sports car driving on the road" \
  --latent_size 32
```
*   生成的圖片會位於 `output/` 資料夾中。

---

## 🛠️ 3. 進階指令 (Advanced Usage)

### 3.1 預處理 (scripts/process_coco.py)
*   `--size`: 原圖解析度。若設為 `512`，訓練時 `img_size` 需改為 `64`。
*   `--ann_file`: 支援 COCO 格式的 JSON 標註檔。

### 3.2 訓練 (scripts/train_transport.py)
*   `--model`: 選擇模型大小 (預設 `SiT-B/2`, 可選 `SiT-XL/2` 等，需修改 config)。
*   `--lr`: 學習率調整。
*   `--epochs`: 對於高品質生成，建議設定 100 以上。

### 3.3 取樣 (scripts/sample.py)
*   `--num_samples`: 一次生成的批次大小。
*   `--latent_size`: **必須**與訓練時的 `--img_size` 一致。

---

## 📁 4. 專案結構

*   `src/`: 模型核心程式碼 (SiT, Transport, VAE Adapter)。
*   `scripts/`: 執行腳本 (Download, Process, Train, Sample)。
*   `data/`: 存放原始與處理後的資料。
*   `checkpoints/`: 存放訓練好的模型權重。
