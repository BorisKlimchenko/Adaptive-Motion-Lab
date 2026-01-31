# ⚡ Adaptive-Motion-Lab

**High-Performance AnimateDiff Pipeline with Hardware-Aware Optimization.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow.svg)](https://huggingface.co/docs/diffusers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

**Adaptive-Motion-Lab** is a production-ready wrapper around the [AnimateDiff](https://github.com/guoyww/AnimateDiff) architecture. It is designed to solve the "configuration hell" of running Latent Diffusion Models across heterogeneous hardware environments (from Google Colab T4 to NVIDIA H100).

Instead of hardcoding settings, this engine uses a **Strategy Pattern** to detect available VRAM and Compute Capability, dynamically injecting the optimal attention mechanisms (SDPA vs xFormers) and VRAM management policies.

## 🏗️ Architecture

The engine implements a **Hardware Abstraction Layer (HAL)** that selects the execution strategy at runtime.

```mermaid
classDiagram
    %% Core Engine Components
    class AdaptiveInferenceEngine {
        +run(scene_name)
        -_load_prompts()
        -_build_pipeline()
    }

    class HardwareProfile {
        +device : str
        +vram_gb : float
        +compute_capability : float
        +is_high_performance_node() bool
    }

    %% Strategy Pattern
    class InferenceStrategy {
        <<Interface>>
        +configure_pipeline()
        +get_resolution_limit()
    }

    class HighPerformanceStrategy {
        +Native SDPA (FlashAttention)
        +Max Resolution: 1024x1024
    }

    class ConsumerStrategy {
        +xFormers Memory Efficient
        +Model CPU Offload
        +VAE Slicing
        +Max Resolution: 512x512
    }

    %% Relationships
    AdaptiveInferenceEngine *-- HardwareProfile : Composition
    AdaptiveInferenceEngine o-- InferenceStrategy : Aggregation
    HardwareProfile --> InferenceStrategy : Determines Factory Output
    InferenceStrategy <|-- HighPerformanceStrategy : Implements
    InferenceStrategy <|-- ConsumerStrategy : Implements
🎬 GalleryDemonstration of Adaptive Inference across different complexity levels (Acts):Act 1: Chaos InitializationAct 2: JEPA Flow StateT4 Optimized (Survival Strategy)A100 Optimized (HighPerf Strategy)<details><summary>👁️ <b>Expand Structural Analysis (Act 3)</b></summary>Act 3: Structural Emergence & Stabilization</details>🚀 Key Features1. Hardware-Aware Dispatch (HAL)The engine automatically profiles the GPU at runtime:Ampere+ (A100, A6000, 3090/4090): Unlocks HighPerformanceStrategy. Uses native PyTorch 2.0 SDPA (F.scaled_dot_product_attention) for maximum throughput. Disables aggressive offloading to keep latencies low.Legacy/Consumer (T4, V100, <16GB VRAM): Activates SurvivalStrategy. Enforces xformers memory-efficient attention, enables model CPU offload, and applies VAE Slicing/Tiling to prevent OOM (Out-of-Memory) errors.2. Deterministic & ReproducibleCross-platform seeding via CPU-based torch.Generator.Strict prompt management via JSON configuration.3. Modular CLI ArchitectureStrategy Pattern: Clean separation of concerns (HardwareProfile -> OptimizationStrategy).CLI Support: Run different experiments using the --prompts argument without changing the source code.🛠 Engineering StackDomainStack & InstrumentationDeep LearningGenerative R&DInfrastructureArchitecture📦 Installation & Usage1. Cloud Execution (Google Colab)For users without high-end local GPUs, use the provided launcher:Open notebooks/Colab_Launcher.ipynb in GitHub.Click the "Open in Colab" button (if available) or download the notebook to Drive.2. Local DevelopmentBash# Clone the repository
git clone [https://github.com/BorisKlimchenko/Adaptive-Motion-Lab.git](https://github.com/BorisKlimchenko/Adaptive-Motion-Lab.git)
cd Adaptive-Motion-Lab

# Install dependencies
pip install -r requirements.txt

# Run Inference (using default config)
python main.py

# Run Inference (using custom config)
python main.py --prompts configs/default_scene.json