import os
import json
import logging
import torch
import numpy as np
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from .dataset.data.json_data import get_json_train_valid_data, generate_and_tokenize_prompt
from .dataset.prompt_maker.translate_prompt_maker import PromptMaker
from functools import partial
from .config import GDIGConfig
from .utils import setup_logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers


class GDIGPipeline:
    """Main pipeline for G-DIG data selection"""
    
    def __init__(self, config: GDIGConfig):
        self.config = config
        
        # Create output directories
        self.clusters_dir = os.path.join(config.output_dir, "clusters")
        self.logs_dir = os.path.join(config.output_dir, "logs")
        os.makedirs(self.clusters_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.logger = setup_logging(config.log_level, config.output_dir)

    
    def run(self) -> None:
        """Run the complete G-DIG pipeline"""
        self.logger.info("Starting G-DIG pipeline")
        
        # Phase I: Compute Hessian matrix
        self.logger.info("Phase I: Computing Hessian matrix")
        self._compute_hessian()
        
        # Phase II: Compute influence scores
        self.logger.info("Phase II: Computing influence scores")
        influence_scores = self._compute_influence_scores()
        
        # Phase III: Clustering and selection
        self.logger.info("Phase III: Clustering and selection")
        selected_indices = self._cluster_and_select(influence_scores)
        
        # Save results
        self.logger.info("Saving results")
        self._save_results(selected_indices, influence_scores)
        
        self.logger.info("Pipeline completed successfully")

    def debug(self) -> None:
        """Run the complete G-DIG pipeline"""
        self.logger.info("Starting G-DIG pipeline")
        
        # Phase I: Compute Hessian matrix
        # self.logger.info("Phase I: Computing Hessian matrix")
        # hessian_path = self._compute_hessian()
        
        # # Phase II: Compute influence scores
        # self.logger.info("Phase II: Computing influence scores")
        # influence_scores = self._compute_influence_scores(hessian_path)
        
        # Phase III: Clustering and selection
        self.logger.info("Phase III: Clustering and selection")
        influence_scores=json.load(open(os.path.join('/home/xp12/cs510/gdig_py/results/gdig_demo/scores.json'), "r"))
        selected_indices = self._cluster_and_select(influence_scores)
        
        # Save results
        self.logger.info("Saving results")
        self._save_results(selected_indices, influence_scores)
        
        self.logger.info("Pipeline completed successfully")
    
    def _compute_hessian(self) -> str:
        """Compute Hessian matrix using kfac_launcher.py"""
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # file dir
            file_dir = os.path.dirname(os.path.abspath(__file__))
            # Get arguments from config
            cmd = ["python3", os.path.join(file_dir, "kfac_launcher.py")] + self.config.get_hessian_args()
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
    
    def _compute_influence_scores(self) -> np.ndarray:
        """Compute influence scores using query_loss_launcher.py"""
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # file dir
            file_dir = os.path.dirname(os.path.abspath(__file__))
            # Get arguments from config
            cmd = ["python3", os.path.join(file_dir, "query_loss_launcher.py")] + self.config.get_influence_args()
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Load and process scores
            scores_path = os.path.join(self.config.output_dir, "scores.json")
            with open(scores_path, "r") as f:
                scores_data = json.load(f)
            
            # Extract scores from results
            # scores = np.array([item["score"] for item in scores_data])
            
            return scores_data
    
    def _cluster_and_select(self, influence_scores: np.ndarray) -> List[int]:
        """Cluster candidates and select diverse examples"""
        
        # Load candidate data for clustering
        with open(self.config.candidate_pool_path, "r") as f:
            candidate_data = json.load(f)
        
        # Compute embeddings for clustering
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_checkpoint, trust_remote_code=True).to(self.config.hessian_device)
        model.eval()

        class tmpQueryargs:
            max_length=256
            use_prompt_loss=False
            prob_data_display=0.1
            data_path=self.config.candidate_pool_path
            valid_data_path=None
            use_large_data=False
            val_set_size=None
            micro_batch_size=4
            tokenizer=self.config.tokenizer_path
            seed=1
        query_data, val_data = get_json_train_valid_data(
            args=tmpQueryargs,
            data_file=tmpQueryargs.data_path,
            valid_data_file=tmpQueryargs.valid_data_path,
            val_set_size=tmpQueryargs.val_set_size,
            prompt_fn=partial(generate_and_tokenize_prompt, args=tmpQueryargs, verbose=False, tokenizer=tokenizer, prompt_maker=PromptMaker(args=tmpQueryargs)),
        )
        queryloader = DataLoader(
            query_data,
            shuffle=False, 
            collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
            batch_size=4, 
            pin_memory=False,
            drop_last=True
        )
        
        embeddings = []
        for example in tqdm(queryloader):
            example = {k: v.to(self.config.hessian_device) for k, v in example.items()}
            with torch.no_grad():
                outputs = model(**example, output_hidden_states=True)
                # Use last hidden state as embedding
                embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
                embeddings.append(embedding)
        
        embeddings = np.vstack(embeddings)
        
        # Perform clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=self.config.num_clusters,
            random_state=42
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Select top examples from each cluster
        selected_indices = []
        for cluster_id in range(self.config.num_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_scores = []
            cluster_indices = np.where(cluster_mask)[0]
            for i in range(len(cluster_indices)):
                cluster_scores.append(influence_scores[cluster_indices[i]]['score'])

            # Select top examples from this cluster
            top_k = max(1, len(cluster_indices) // 10)  # Select top 10% from each cluster
            top_indices = cluster_indices[np.argsort(cluster_scores)[-top_k:]]
            selected_indices.extend(top_indices)
        
        return selected_indices
    
    def _save_results(self, selected_indices: List[int], influence_scores: np.ndarray) -> None:
        """Save pipeline results"""
        # Load candidate data
        with open(self.config.candidate_pool_path, "r") as f:
            candidate_data = json.load(f)
        
        # Save selected examples
        selected_examples = [candidate_data[i] for i in selected_indices]
        with open(os.path.join(self.config.output_dir, "selected.json"), "w") as f:
            json.dump(selected_examples, f, indent=2, ensure_ascii=False)
        
        # Save influence statistics
        all_scores = np.array([influence_scores[i]['score'] for i in selected_indices])
        influence_stats = pd.DataFrame({
            "mean": all_scores.mean(),
            "min": all_scores.min(),
            "max": all_scores.max()
        }, index=[0])
        influence_stats.to_csv(os.path.join(self.config.output_dir, "influence.csv"))
        
        # Save logs
        self.logger.info(f"Selected {len(selected_indices)} examples")
        self.logger.info(f"Results saved to {self.config.output_dir}") 