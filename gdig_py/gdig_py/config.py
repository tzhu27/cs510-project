import os
import yaml
from dataclasses import dataclass
from typing import Optional, Literal, List
from pathlib import Path

@dataclass
class GDIGConfig:
    """Configuration for G-DIG pipeline"""
    
    # Data paths
    seed_set_path: str
    candidate_pool_path: str
    model_checkpoint: str
    tokenizer_path: str  # Path to tokenizer (can be same as model_checkpoint)
    
    # Hessian computation parameters (from kfac_launcher.py)
    hessian_device: str = "cuda:0"  # -g
    num_gpus: int = 8  # -n
    hessian_trials: int = 1  # -t
    hessian_output: str = "hessian.pkl"  # -o
    
    # Influence score parameters (from query_loss_launcher.py)
    influence_batch_size: int = 2  # -bq
    influence_lambda: float = 0.5  # -lmd
    full_score: bool = False  # --full-score
    use_ekfac: bool = False  # --ekfac
    start_index: Optional[int] = None  # --start
    end_index: Optional[int] = None  # --end
    layer_type: str = "b"  # --layer
    
    # Clustering parameters
    num_clusters: int = 50
    cluster_metric: Literal["cosine", "euclidean"] = "cosine"
    
    # General settings
    output_dir: str = "results/gdig_run1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    visualize: bool = False
    
    # Model-specific settings
    model_type: Literal["bloom", "llama", "baichuan"] = "bloom"
    max_length: int = 256
    micro_batch_size: int = 4
    use_prompt_loss: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GDIGConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert paths to absolute paths
        for key in ["seed_set_path", "candidate_pool_path", 
                    "output_dir"]:
            if key in config_dict:
                config_dict[key] = str(Path(config_dict[key]).resolve())
        
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration values"""
        # Check if files exist
        for path_key in ["seed_set_path", "candidate_pool_path"]:
            path = getattr(self, path_key)
            if not os.path.exists(path):
                raise ValueError(f"File not found: {path}")
        
        # Validate numeric parameters
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        if self.hessian_trials <= 0:
            raise ValueError("hessian_trials must be positive")
        if self.influence_batch_size <= 0:
            raise ValueError("influence_batch_size must be positive")
        if self.influence_lambda <= 0:
            raise ValueError("influence_lambda must be positive")
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        
        # Validate indices
        if self.start_index is not None and self.end_index is not None:
            if self.start_index > self.end_index:
                raise ValueError("start_index must be less than or equal to end_index")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "factor"), exist_ok=True)
    
    def get_hessian_args(self) -> List[str]:
        """Get arguments for kfac_launcher.py"""
        return [
            "-g", self.hessian_device,
            "-n", str(self.num_gpus),
            "-d", self.candidate_pool_path,
            "-o", os.path.join(self.output_dir, self.hessian_output),
            "-m", self.model_checkpoint,
            "-t", str(self.hessian_trials),
            "-k", self.tokenizer_path,
            "-c", os.path.join(self.output_dir, "factor")
        ]
    
    def get_influence_args(self) -> List[str]:
        """Get arguments for query_loss_launcher.py"""
        args = [
            "-n", str(self.num_gpus),
            "-k", os.path.join(self.output_dir, self.hessian_output),
            "-m", self.model_checkpoint,
            "-t", self.tokenizer_path,
            "-o", os.path.join(self.output_dir, "scores.json"),
            "-d", self.candidate_pool_path,
            "-q", self.seed_set_path,
            "-bq", str(self.influence_batch_size),
            "-lmd", str(self.influence_lambda),
            "-c", os.path.join(self.output_dir, "factor")
        ]
        
        # Add optional arguments
        if self.full_score:
            args.append("--full-score")
        if self.use_ekfac:
            args.append("--ekfac")
        if self.start_index is not None:
            args.extend(["--start", str(self.start_index)])
        if self.end_index is not None:
            args.extend(["--end", str(self.end_index)])
        if self.layer_type:
            args.extend(["--layer", self.layer_type])
        
        return args 