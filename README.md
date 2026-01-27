# JEPA-Synthetic-Lab 🧬

> **Official implementation of "JEPA vs LLM" visualization pipeline.**
> Audio-reactive Latent Space navigation using AnimateDiff & Stable Diffusion.

This repository contains the **SMA-01 Core Engine**, a generative pipeline designed to visualize abstract concepts of machine intelligence (Joint Embedding Predictive Architecture vs. Large Language Models).

## 🚀 Key Features

* **Adaptive Compute Strategy:** The engine automatically detects your hardware (`A100/A6000` vs `T4/Home GPU`) and switches between *High Performance* and *Survival Mode* (CPU Offloading).
* **Physics-Based Rendering:** Uses `EulerDiscreteScheduler` to simulate momentum in latent space sampling.
* **JSON-Driven Scenes:** All generation parameters are controlled via `prompts.json` for reproducible experiments.
* **Universal Auth:** Seamlessly handles Hugging Face tokens via Environment Variables or Google Colab Userdata.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BorisKlimchenko/JEPA-Synthetic-Lab.git](https://github.com/BorisKlimchenko/JEPA-Synthetic-Lab.git)
    cd JEPA-Synthetic-Lab
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration (`prompts.json`)

The logic is separated from the configuration. Edit `prompts.json` to define your scenes.

| Parameter | Description |
| :--- | :--- |
| `motion_scale` | Controls the "volatility" of the video (Higher = more chaos). |
| `base_steps` | Number of inference steps (automatically adjusted by motion complexity). |
| `seed` | Lock seed for reproducibility (`-1` for random). |

**Example structure:**
```json
"Act_1_Chaos": {
  "description": "Dark Ocean of Noise (LLM State)",
  "positive": "macro shot of boiling black ferrofluid...",
  "motion_scale": 1.4
}