# SewingSiT: 將 SiT 轉換為文字條件生成模型

## 更新摘要 (v0.1)
我們已成功修改了原始的 SiT (Scalable Interpolant Transformers) 架構，使其支援文字條件 (Text Conditioning)，讓它能像 DiT 或 SDXL 一樣執行文生圖 (Text-to-Image) 任務。

### `models.py` 的關鍵修改
1.  **移除 `LabelEmbedder`**:
    *   移除了原始用於 ImageNet 的類別條件嵌入層 (Class-Conditional Embedding)。
2.  **新增 `CrossAttention` 層**:
    *   實作了標準的多頭交叉注意力機制 (Multi-Head Cross Attention) 以注入外部內容。
    *   輸入：圖片 Tokens (Query) + 文字嵌入向量 (Key/Value)。
3.  **更新 `SiTBlock`**:
    *   將 `CrossAttention` 整合進 Transformer 區塊結構中。
    *   新結構順序：`Self-Attn -> Cross-Attn (新增) -> MLP`。
4.  **修改 `SiT` 主類別**:
    *   `forward` 函數簽名已從 `(x, t, y)` 變更為 `(x, t, context)`。
    *   `context` 預期接收形狀為 `(N, Sequence_Length, Hidden_Dim)` 的張量 (例如 CLIP 的文字嵌入)。
5.  **輸出通道調整**:
    *   已修正 `forward` 傳遞過程，現在若設定 `learn_sigma=True`，模型會正確回傳完整的 `in_channels * 2` (Mean + Variance)，這對於計算 Loss 至關重要。

## 如何測試
1.  **環境設定**: 確保您位於 `sewingSiT` 環境中，並已安裝 PyTorch。
2.  **執行測試腳本**:
    ```bash
    python scripts/test_sit_text.py
    ```
3.  **預期輸出**:
    *   這將驗證模型是否能正確實例化並進行一次 Forward Pass。
    *   若看到 `Success!` 且輸出形狀正確 (例如 `[2, 8, 32, 32]`)，即代表架構修改成功。

## 資料流程 (Data Pipeline)

為了訓練模型，我們需要將圖片與文字轉換為模型可讀的格式 (Latents + Embeddings)。

### 1. 生成測試用假資料 (Dummy Data)
如果您沒有現成的資料集，可以使用以下腳本生成隨機圖片與對應的文字描述：
```bash
python scripts/generate_dummy_data.py --data_dir data/raw_images
```

### 2. 資料預處理 (Preprocessing)
將原始圖片編碼為 Latents，並將文字轉為 CLIP Embeddings，存為 `.npz` 格式以加速訓練：
```bash
python scripts/preprocess_data.py --data_dir data/raw_images --output_dir data/processed
```
*   **Latents**: 使用 SSD-1B/SDXL VAE (1024x1024 -> 128x128x4)
*   **Text Embeddings**: 使用 CLIP ViT-L/14 (77x768)

## 訓練 (Training)

我們提供了兩個版本的訓練腳本：

### 1. 基礎驗證 (`train_simple.py`)
驗證「載入資料 -> 模型運算 -> 計算 MSE Loss -> 反向傳播」的完整流程。
```bash
python scripts/train_simple.py
```

### 2. 進階訓練 (`train_transport.py`)
整合 SiT 核心的 **Transport (Rectified Flow / Velocity Matching)** 機制與 **WandB** 監控。這是訓練 SOTA 生成模型的標準配置。
```bash
# 測試執行
python scripts/train_transport.py --epochs 5

# 正式訓練 (需登入 WandB)
python scripts/train_transport.py --use_wandb --run_name "sewing-sit-v1"
```

## COCO Dataset Workflow

如果您下載了 MS-COCO 資料集，可使用專用腳本進行處理：

1.  **下載資料**:
    ```bash
    python scripts/download_coco.py --target_dir /mnt/f/nccu/project/data/2dImg --split test_small
    ```
2.  **解壓縮** (腳本會自動處理，或手動解壓至 `.../val2017` 等)。
3.  **預處理** (影像轉 Latents, 文字轉 Embeddings):
    ```bash
    python scripts/process_coco.py \
      --img_dir /mnt/f/nccu/project/data/2dImg/val2017 \
      --ann_file /mnt/f/nccu/project/data/2dImg/annotations/captions_val2017.json \
      --output_dir /mnt/f/nccu/project/data/2dImg/processed_val \
      --size 256
    ```
4.  **訓練** (使用真實資料):
    ```bash
    python scripts/train_transport.py \
      --data_dir /mnt/f/nccu/project/data/2dImg/processed_val \
      --epochs 50 \
      --batch_size 32 \
      --use_wandb \
      --run_name "coco-val-256-test"
    ```

5.  **生成測試 (Sampling)**:
    ```bash
    python scripts/sample.py \
      --checkpoint checkpoints/run001-coco-val-32.pt \
      --prompt "A red sports car" \
      --latent_size 32
    ```
    *   生成的圖片會存在 `output/` 資料夾中。
    *   *注意：由於目前僅使用少量驗證集 (5k images) 進行短期訓練，生成結果可能較為抽象，主要用於驗證程式碼流程是否打通。*

## 未來工作 (Future Work)
1.  **擴充資料規模**: 改用 MS-COCO 完整訓練集 (118k images) 或 Laion-400M 等大規模資料集。
2.  **增加訓練時長**: 文生圖模型通常需要長時間訓練 (如 100+ epochs) 才能看到清晰語義。
3.  **解析度提升**: 將訓練解析度提升回 512 或 1024。
4.  **雙 Text Encoder**: 參考 SDXL/SSD-1B，同時引入 OpenCLIP ViT-bigG 與 CLIP ViT-L，增強對提示詞的理解能力。
5.  **模型規模擴展**: 嘗試訓練更大的 `SiT-XL/2` (參數量類似 DiT-XL) 以獲得更好的生成品質。
