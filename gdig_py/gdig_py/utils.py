import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from gdig_py.nngeometry.object import PMatEKFAC
import json
import pandas as pd

def setup_logging(log_level: str, output_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("gdig")
    logger.setLevel(getattr(logging, log_level))
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(output_dir, "logs", "gdig.log"))
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def compute_influence_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    seed_data: List[Dict[str, Any]],
    candidate_data: List[Dict[str, Any]],
    pF: PMatEKFAC,
    batch_size: int,
    damping: float
) -> np.ndarray:
    """Compute influence scores for all candidates"""
    device = next(model.parameters()).device
    model.eval()
    
    # Compute gradients for seed examples
    seed_gradients = []
    for example in seed_data:
        inputs = tokenizer(
            example["text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.set_grad_enabled(True):
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            
            # Get gradients
            grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            seed_gradients.append(grad.cpu().numpy())
            
            # Clear gradients
            model.zero_grad()
    
    seed_gradients = np.stack(seed_gradients)
    
    # Compute influence scores for candidates
    influence_scores = []
    for i in range(0, len(candidate_data), batch_size):
        batch = candidate_data[i:i + batch_size]
        batch_scores = []
        
        for example in batch:
            inputs = tokenizer(
                example["text"],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            with torch.set_grad_enabled(True):
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                
                # Get gradients
                grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                
                # Compute influence score
                score = -np.dot(seed_gradients, grad.cpu().numpy())
                batch_scores.append(score)
                
                # Clear gradients
                model.zero_grad()
        
        influence_scores.extend(batch_scores)
    
    return np.array(influence_scores)

def save_results(
    output_dir: str,
    selected_examples: List[Dict[str, Any]],
    influence_scores: np.ndarray,
    cluster_assignments: np.ndarray
) -> None:
    """Save pipeline results to disk"""
    # Save selected examples
    with open(os.path.join(output_dir, "selected.json"), "w") as f:
        json.dump(selected_examples, f, indent=2)
    
    # Save influence statistics
    influence_stats = pd.DataFrame({
        "mean": influence_scores.mean(axis=1),
        "min": influence_scores.min(axis=1),
        "max": influence_scores.max(axis=1)
    })
    influence_stats.to_csv(os.path.join(output_dir, "influence.csv"))
    
    # Save cluster assignments
    cluster_df = pd.DataFrame({
        "example_id": range(len(cluster_assignments)),
        "cluster": cluster_assignments
    })
    cluster_df.to_csv(os.path.join(output_dir, "clusters", "assignments.csv")) 