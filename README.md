# JEPA-Synthetic-Lab 🧬

[![SMA-01 Core](https://img.shields.io/badge/Core_Engine-SMA--01_v3.1-blue?style=flat-square)](https://github.com/BorisKlimchenko)
[![Architecture](https://img.shields.io/badge/Architecture-Latent_Diffusion-orange?style=flat-square)](https://huggingface.co/docs/diffusers)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)

> **Official implementation of "JEPA vs LLM" visualization pipeline.**
> Deterministic Latent Space navigation using AnimateDiff & Semantic Motion Adapters.

This repository hosts the **SMA-01 Core Engine**, a generative pipeline designed to visualize abstract concepts of machine intelligence. It serves as a bridge between abstract cognitive theories (Joint Embedding Predictive Architecture) and concrete visual embeddings.

## 🚀 Key Features

* **Adaptive Compute Strategy:** The engine performs real-time heuristics on your GPU topology. It automatically switches between `HighPerformance` (A100/A6000) and `SurvivalMode` (T4/CPU Offloading).
* **Physics-Based Rendering:** Implements `EulerDiscreteScheduler` to ensure conservation of momentum during latent sampling.
* **Deterministic Integrity:** "God plays dice, but we don't." Full seed control via `prompts.json` for reproducible experiments.
* **Motion-Reactive Logic:** The `motion_scale` parameter dynamically modulates inference steps to stabilize high-entropy scenes.
* **Universal Auth:** Seamlessly handles Hugging Face tokens via Environment Variables (Docker) or Google Colab Userdata.

## 🛠️ Installation

**1. Clone the Protocol:**
```bash
git clone [https://github.com/BorisKlimchenko/JEPA-Synthetic-Lab.git](https://github.com/BorisKlimchenko/JEPA-Synthetic-Lab.git)
cd JEPA-Synthetic-Lab
