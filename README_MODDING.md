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

我們提供了互動式選單 `main.py`，這是最推薦的操作方式。

```bash
python main.py
```

### 📋 選單功能詳解

進入選單後，您會看到以下選項。請依照您的需求選擇：

#### 1. 📥 Download Data (資料下載)
*   **Target Directory**: 資料存放位置 (預設: `../data/2dImg/coco_2017`)
*   **Split**:
    *   `test_small` (推薦): 下載 COCO 驗證集 (5k 張圖, ~1GB)，適合快速測試。
    *   `train`: 下載完整 COCO 訓練集 (118k 張圖, ~18GB)，適合正式訓練。

#### 2. 🔄 Process Data (資料預處理)
將圖片與文字轉換為模型訓練用的 Latents 與 Embeddings（備料階段）。
*   **Image Directory**: 原始圖片資料夾 (例如 `.../val2017` 或 `.../train2017`)。
*   **Output Directory**: 處理後的 `.npz` 存放處。
*   **Image Resolution**: 圖片解析度 (預設 `256`)。若顯卡記憶體夠大 (24GB+)，可設為 `512`。

#### 3. 🏋️ Train Model (模型訓練)
*   **Processed Data Directory**: 指向剛剛 Process 出來的資料夾。
*   **Epochs**: 訓練輪數 (測試建議 `10`，正式建議 `50`~`100`+)。
*   **Batch Size**: 批次大小 (依顯存而定，3090/4090 可開 `32`~`64`)。
*   **Latent Size**: 必須對應解析度 (解析度/8)。例如 256 對應 `32`，512 對應 `64`。
*   **Use WandB**: 是否使用 Weights & Biases 監控訓練曲線 (推薦 `y`)。

#### 4. 🎨 Generate Image (生成圖片)
*   **Select Checkpoint**: 系統會自動列出 `checkpoints/` 下的模型讓您選擇。
*   **Prompt**: 輸入您想生成的英文描述 (例如 "A futuristic city with flying cars")。
*   **Latent Size**: **必須**與訓練時的設定一致 (例如 `32`)。

---

### 💡 操作範例 (Scenarios)

#### 場景 A: 新手上路 (快速驗證流程)
目標：使用少量資料確認環境與程式碼沒問題。
1.  **Download**: Split 輸入 `test_small`。
2.  **Process**: Image Resolution 輸入 `256`。
3.  **Train**: Epochs 輸入 `10`，Latent Size 輸入 `32`。
4.  **Generate**: 選擇剛剛訓練好的 checkpoint，Prompt 隨意輸入，Latent Size 輸入 `32`。

#### 場景 B: 火力全開 (訓練高品質模型)
目標：使用完整資料集訓練更強的模型。
1.  **Download**: Split 輸入 `train` (需準備 30GB+ 硬碟空間)。
2.  **Process**: 針對 `train2017` 進行處理。Image Resolution 可嘗試 `512` (若硬碟與顯存允許)。
3.  **Train**: Epochs 設定 `100`，Latent Size 設定 `64` (對應 512 解析度)。
4.  **Generate**: 享受您的訓練成果！

---

如果偏好手動執行指令 (Advanced)，請參考下方步驟：
我們使用 MS-COCO 驗證集 (約 1GB) 進行快速測試。腳本會自動下載圖片、標註，並將其轉換為訓練用的 Latents (`.npz`)。

```bash
# 1. 下載 COCO Val2017
python scripts/download_coco.py --target_dir ../data/2dImg/coco_2017 --split test_small

# 2. 預處理 (圖片尺寸 256x256 -> Latent 32x32)
python scripts/process_coco.py \
  --img_dir ../data/2dImg/coco_2017/val2017 \
  --ann_file ../data/2dImg/coco_2017/annotations/captions_val2017.json \
  --output_dir ../data/2dImg/coco_2017/processed_val \
  --size 256
```

### Step 2: 訓練模型
使用 Rectified Flow (Transport) 進行訓練。

```bash
python scripts/train_transport.py \
  --data_dir ../data/2dImg/coco_2017/processed_val \
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
