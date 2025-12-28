# 專案報告: SewingSiT - 將 SiT 轉型為開放詞彙圖像生成器

## 1. 專案概述 (Project Overview)
**SewingSiT** 是一個基於 Scalable Interpolant Transformers (SiT) 架構的修改實作。我們的主要目標是將原本僅限於類別條件生成 (如 ImageNet 固定分類) 的 SiT，轉型為具備文字條件控制 (Text-Conditional) 能力的模型，使其能夠執行開放詞彙的文生圖 (Text-to-Image) 任務。

這項改造將 SiT 的架構提升至與 Stable Diffusion 3 (SD3) 和 PixArt-Alpha 等現代 SOTA 模型相近的層次，並在此基礎上利用了 Rectified Flow (Transport) 訓練的高效率優勢。

## 2. 關鍵架構修改 (The "Modding")

我們透過「縫合 (Sewing)」現代組件至原始 SiT 骨幹中，成功完成了模型的轉型：

### A. 從分類標籤到文字嵌入 (Text Embeddings)
*   **移除部分**: `LabelEmbedder`。模型不再接收離散的類別 ID (例如 `y=10` 代表鴕鳥)。
*   **新增部分**: 在每個 `SiTBlock` 中植入 `CrossAttention` 層。
*   **機制變更**:
    *   原始結構: `Self-Attention -> MLP` (透過 adaLN 進行條件控制)。
    *   **修改後結構**: `Self-Attention -> Cross-Attention -> MLP`。
    *   新加入的 `CrossAttention` 層使模型能夠關注並理解由 Text Encoder 提供的文字嵌入序列。

### B. 整合 SSD-1B 生態系
為了實現高解析度合成與豐富的語義理解，我們整合了 **SSD-1B (Segmind Stable Diffusion 1B)** 生態系中的關鍵組件：
1.  **VAE (變分自動編碼器)**:
    *   我們直接利用 **SSD-1B VAE** 將圖片壓縮為潛在特徵 (Latents)。
    *   **作用**: 將 256x256 或 1024x1024 的 RGB 圖片壓縮成 32x32 或 128x128 的張量 (8倍下採樣)。這對於降低 Transformer 的運算負載至關重要。
2.  **Text Encoder (文字編碼器)**:
    *   我們採用了 **CLIP ViT-L/14** (SDXL/SSD-1B 的標準配置) 來將文字提示詞編碼為 768 維的嵌入向量。

### 4. 架構示意圖 (Architecture Diagrams)

為了讓讀者更清楚專案的「縫合」過程與架構變遷，以下展示三個階段的架構示意圖：

#### 1. 原始 SiT (Original SiT)
原始 SiT 設計用於類別條件生成 (如 ImageNet)。它使用 `LabelEmbedder` 將離散的類別 ID 轉換為嵌入向量，並與時間嵌入相加後，透過 `adaLN` (Adaptive Layer Norm) 來調節 Transformer Block。其內部只有 Self-Attention。

```mermaid
graph TD
    subgraph Original_SiT [Original SiT (Class Conditional)]
        direction TB
        Input_Img[Input Image / Latent] --> PatchEmbed
        Class_Label[Class Label y] --> LabelEmb[LabelEmbedder]
        Time_Step[Timestep t] --> TimeEmb[TimestepEmbedder]
        
        PatchEmbed --> Block_Start
        
        LabelEmb --> C_Add((+))
        TimeEmb --> C_Add
        C_Add --> Condition_Vector[Condition Vector c]
        
        subgraph SiT_Block [SiT Transformer Block]
            direction TB
            Input_Block[Input x] --> Norm1
            Condition_Vector -.->|adaLN Modulation| Norm1
            Norm1 --> SelfAttn[Self-Attention]
            SelfAttn --> Add1((+))
            Input_Block --> Add1
            
            Add1 --> Norm2
            Condition_Vector -.->|adaLN Modulation| Norm2
            Norm2 --> MLP
            MLP --> Add2((+))
            Add1 --> Add2
            Add2 --> Output_Block[Output x]
        end
        
        Block_Start --> SiT_Block
        SiT_Block --> Final_Layer
        Final_Layer --> Output_Img[Generated Image]
    end
```

#### 2. SSD-1B (Segmind Stable Diffusion 1B)
SSD-1B 是基於 SDXL 的縮小版，但保留了強大的文字理解能力。其核心特徵是使用兩組 Text Encoder (OpenCLIP ViT-L/14 和 ViT-G/14) 提取文字特徵，並透過 **Cross-Attention** 將這些特徵注入到 UNet (或 Transformer) 中。

```mermaid
graph TD
    subgraph SSD_1B [SSD-1B Ecosystem]
        direction TB
        Text_Prompt[Text Prompt] --> TextEnc1[CLIP ViT-L/14]
        Text_Prompt --> TextEnc2[CLIP ViT-G/14]
        
        TextEnc1 --> Text_Embeds[Text Embeddings]
        TextEnc2 --> Text_Embeds
        
        Input_Latent[Noisy Latent] --> VAE_Enc
        
        subgraph Backbone [Diffusion Backbone (UNet/Transformer)]
            direction TB
            Layer_In --> SA[Self-Attention]
            SA --> CA[Cross-Attention]
            Text_Embeds -.->|Key/Value| CA
            CA --> FFN[FeedForward]
            FFN --> Layer_Out
        end
        
        Input_Latent --> Backbone
        Backbone --> VAE_Dec[VAE Decoder]
        VAE_Dec --> Gen_Img[Generated Image]
    end
```

#### 3. SewingSiT (Finished Product)
SewingSiT 是我們將上述兩者結合的成果。我們移除了 `LabelEmbedder`，保留了 `TimeEmbedder` 用於 `adaLN`，並在每個 Block 中 Self-Attention 之後「縫入」了 `Cross-Attention` 層。這個 Cross-Attention 專門用來接收來自 SSD-1B 生態系 (CLIP ViT-L/14) 的文字嵌入，實現了文生圖功能。

```mermaid
graph TD
    subgraph Sewing_SiT [SewingSiT (Text Conditional)]
        direction TB
        style Sewing_SiT fill:#e1f5fe,stroke:#01579b,stroke-width:2px
        
        Input_Noise[Input Noise (Latent)] --> Patch_Embed[PatchEmbed]
        Text_Prompt[Text Prompt] -->|Pre-computed| CLIP[CLIP ViT-L/14]
        CLIP --> Text_Context[Text Context (N, 77, 768)]
        
        Time_Step[Timestep t] --> Time_Emb[TimestepEmbedder]
        Time_Emb --> Ada_Cond[adaLN Condition c]
        
        Patch_Embed --> Block_In
        
        subgraph Modified_Block [Modified SiT Block]
            style Modified_Block fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
            
            Block_Input[Input x] --> Norm_1[Norm1 (adaLN)]
            Ada_Cond -.->|Modulate| Norm_1
            Norm_1 --> Self_Attn[Self-Attention]
            Self_Attn --> Add_1((+))
            Block_Input --> Add_1
            
            Add_1 --> Norm_Cross[Norm (Layernorm)]
            Norm_Cross --> Cross_Attn[Cross-Attention]
            Text_Context -.->|Key/Value| Cross_Attn
            Cross_Attn --> Add_Cross((+))
            Add_1 --> Add_Cross
            
            Add_Cross --> Norm_2[Norm2 (adaLN)]
            Ada_Cond -.->|Modulate| Norm_2
            Norm_2 --> MLP[MLP]
            MLP --> Add_2((+))
            Add_Cross --> Add_2
            Add_2 --> Block_Out[Output x]
        end
        
        Block_In --> Modified_Block
        Modified_Block --> Final_Layer[Final Layer]
        Final_Layer --> VAE_Dec[SSD-1B VAE Decoder]
        VAE_Dec --> Output_Image[Generated Image]
    end
```

## 3. 實作成果 (Implementation Achievements)

### 1. 資料管道 (Data Pipeline)
*   **COCO 資料集支援**: 實作了自動下載與處 MS-COCO 資料集的腳本。
*   **預處理機制**: 建立了 `scripts/process_coco.py`，預先計算 Latents (透過 SSD-1B VAE) 與 Text Embeddings (透過 CLIP)。這透過移除訓練時的即時編碼需求，大幅提升了訓練速度。

### 2. 訓練管道 (Training Pipeline)
*   **Rectified Flow (Transport)**: 成功整合了 `transport` 函式庫。
*   **速度匹配 (Velocity Matching)**: 模型被訓練來預測將雜訊傳輸至資料的「速度場」，這種方法以能用比傳統擴散模型更少的步數生成高品質圖片而聞名。
*   **WandB 監控**: 驗證了 Loss 追蹤與訓練穩定性。

### 3. 取樣與推論 (Sampling & Inference)
*   **ODE 解算器**: 實作了 `scripts/sample.py`，使用 `dopri5` ODE 解算器來逆轉流向，從純雜訊中生成符合文字描述的圖片。
*   **端對端閉環**: 驗證了完整的循環：`隨機雜訊 -> SiT + Transport -> Latent -> VAE Decoder -> 圖片`。

## 4. 當前狀態 (Current Status)
*   **模型**: SiT-B/2 (已修改加入 Cross-Attention)。
*   **訓練資料**: 已在 COCO 驗證集 (小規模，5000 張圖片) 上完成驗證。
*   **能力**: 具備執行完整訓練與推論流程的能力。由於目前訓練時間較短且數據量較小，模型目前生成的結果可能較為抽象，但整套軟體管線 (Pipeline) 已被證實功能完整且無錯誤。

## 5. 下一步 (Next Steps)
*   **擴大數據**: 轉移至完整的 COCO 訓練集 (118k) 或 Laion-400M 進行訓練。
*   **擴大訓練**: 將訓練 Epochs 從 10 提升至 100+。
*   **擴大模型**: 嘗試訓練參數更多的 `SiT-XL/2`。
