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

我們提供了一個精簡版的訓練腳本 `scripts/train_simple.py` 來驗證「載入資料 -> 模型運算 -> 計算 Loss -> 反向傳播」的完整流程。

### 執行測試訓練
```bash
python scripts/train_simple.py
```
此腳本會：
1.  讀取 `data/processed` 中的 `.npz` 訓練資料。
2.  初始化 `SiT-B/2` 文字條件模型。
3.  執行 5 個 Epoch 的模擬訓練。
4.  將權重儲存至 `checkpoints/sit_text_test.pt`。
