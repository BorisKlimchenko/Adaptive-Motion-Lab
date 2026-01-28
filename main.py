import torch
import json
import os
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

# Third-party Libs
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

# --- 1. INSTRUMENTATION (LOGGING) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [SMA-01 CORE] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION (IMMUTABLE) ---
@dataclass(frozen=True)
class EngineConfig:
    """
    Immutable Configuration for JEPA-Synthetic-Lab Environment.
    Follows SOLID principles.
    """
    base_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    prompts_path: str = "prompts.json" # Local path
    output_dir: str = "renders"
    default_negative: str = "bad quality, worse quality, low resolution"

# --- 3. STRATEGY PATTERN (ADAPTIVE COMPUTE) ---
class OptimizationStrategy(ABC):
    @abstractmethod
    def apply(self, pipe: AnimateDiffPipeline):
        pass

class HighPerformanceStrategy(OptimizationStrategy):
    """A100/A6000 Class: Full VRAM utilization."""
    def apply(self, pipe: AnimateDiffPipeline):
        logger.info("🚀 Strategy: HIGH PERFORMANCE. All tensors loaded to VRAM.")
        # Optional: pipe.enable_xformers_memory_efficient_attention()

class SurvivalStrategy(OptimizationStrategy):
    """T4/Home GPU Class: Aggressive offloading."""
    def apply(self, pipe: AnimateDiffPipeline):
        logger.info("🛡️ Strategy: SURVIVAL MODE. CPU Offloading active.")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

def detect_hardware_strategy() -> OptimizationStrategy:
    """Heuristic analysis of the compute manifold."""
    if not torch.cuda.is_available():
        logger.warning("⚠️ CPU DETECTED. Latent navigation will be extremely slow.")
        return SurvivalStrategy()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    device_name = torch.cuda.get_device_name(0)
    
    logger.info(f"🖥️ Hardware Detected: {device_name} ({vram_gb:.1f} GB)")

    if vram_gb > 20.0:
        return HighPerformanceStrategy()
    else:
        return SurvivalStrategy()

# --- 4. CORE ENGINE ---
class LatentMotionEngine:
    """
    SMA-01 Core Engine v3.1
    Orchestrates the generative pipeline with physics-based constraints.
    """
    
    def __init__(self):
        logger.info("⚙️ Initializing SMA-01 Core Engine...")
        
        self.config = EngineConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # 1. Credential Injection
        self.token = self._resolve_credentials()
        
        # 2. Load Vector Database (Prompts)
        self.prompts_db = self._load_prompts()
        
        # 3. Strategy Selection
        self.strategy = detect_hardware_strategy()
        
        # 4. Pipeline Construction
        self.pipe = self._build_pipeline()

    def _resolve_credentials(self) -> Optional[str]:
        """
        Universal Token Resolver (Docker + Colab + Local).
        """
        # Priority A: Env Variable (Production/Docker)
        token = os.getenv("HF_TOKEN")
        if token:
            logger.info("🔑 Token loaded from Environment.")
            return token
            
        # Priority B: Colab Userdata (Sandbox)
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
            if token:
                logger.info("🔑 Token loaded from Colab Userdata.")
                return token
        except ImportError:
            pass

        logger.warning("⚠️ HF_TOKEN not found. Accessing public manifold only.")
        return None

    def _load_prompts(self) -> Dict[str, Any]:
        if not os.path.exists(self.config.prompts_path):
            logger.error(f"❌ Critical: Config {self.config.prompts_path} missing.")
            return {"scenes": {}}
            
        with open(self.config.prompts_path, 'r') as f:
            data = json.load(f)
        logger.info("Correction Vectors (Prompts) loaded.")
        return data

    def _build_pipeline(self) -> AnimateDiffPipeline:
        logger.info("🔌 Mounting Neural Adapters...")
        
        adapter = MotionAdapter.from_pretrained(
            self.config.motion_adapter_id,
            torch_dtype=self.dtype,
            token=self.token
        )

        pipe = AnimateDiffPipeline.from_pretrained(
            self.config.base_model_id,
            motion_adapter=adapter,
            torch_dtype=self.dtype,
            token=self.token
        )

        # Physics Scheduler (Euler Ancestral implies conservation of momentum in sampling)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing", 
            beta_schedule="linear"
        )

        self.strategy.apply(pipe)
        return pipe

    def calculate_compute_cost(self, base_steps: int, motion_scale: float) -> int:
        """
        Calculates required inference steps based on motion complexity.
        Higher motion = higher entropy = more steps needed for convergence.
        """
        if motion_scale > 1.0:
            return int(base_steps * 1.2) # +20% compute for high motion
        return base_steps

    def render(self, scene_name: str, forced_seed: int = -1) -> str:
        """
        Executes the generative trajectory.
        """
        # A. Validation
        if scene_name not in self.prompts_db.get('scenes', {}):
            raise ValueError(f"❌ Scene '{scene_name}' undefined in Latent Space.")
            
        scene_data = self.prompts_db['scenes'][scene_name]
        sys_config = self.prompts_db.get('system_settings', {})

        # B. Parameter Extraction
        width = sys_config.get('width', 512)
        height = sys_config.get('height', 512)
        fps = sys_config.get('fps', 8)
        base_steps = sys_config.get('base_steps', 25)
        
        # C. Latent Physics Injection (Motion Scale)
        motion_scale = scene_data.get('motion_scale', 1.0)
        inference_steps = self.calculate_compute_cost(base_steps, motion_scale)

        # D. Seed Integrity
        # JSON seed overrides random, forced_seed overrides JSON.
        json_seed = scene_data.get('seed', -1)
        
        if forced_seed != -1:
            seed = forced_seed
        elif json_seed != -1:
            seed = json_seed
        else:
            seed = random.randint(0, 2**32 - 1)

        logger.info(f"🎲 Seed Locked: {seed}")
        
        # <<< CRITICAL FIX: Generator must be on CPU for cross-platform reproducibility >>>
        # Was: generator = torch.Generator(self.device).manual_seed(seed)
        generator = torch.Generator("cpu").manual_seed(seed)

        # E. Execution
        logger.info(f"🎬 Action: {scene_name}")
        logger.info(f"   Context: {scene_data.get('description')}")
        logger.info(f"   Physics: Motion Scale {motion_scale} -> Steps {inference_steps}")

        output = self.pipe(
            prompt=scene_data['positive'],
            negative_prompt=scene_data.get('negative', self.config.default_negative),
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=inference_steps,
            generator=generator,
            width=width,
            height=height,
        )
        
        # F. Artifact Storage
        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"{self.config.output_dir}/{scene_name}_{seed}.gif"
        export_to_gif(output.frames[0], filename)
        
        logger.info(f"💾 Artifact materialized: {filename}")
        return filename

# --- 5. ENTRY POINT ---
if __name__ == "__main__":
    print("--- SMA-01 CORE INITIALIZED ---")
    try:
        engine = LatentMotionEngine()
        
        # Example: Render all scenes defined in JSON
        for scene_key in engine.prompts_db['scenes'].keys():
            engine.render(scene_key)
            
        print("✅ Batch Processing Complete.")
    except Exception as e:
        logger.critical(f"❌ System Failure: {e}")
        raise