"""
Main module for the G-DIG algorithm.
"""
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import transformers
from tqdm import tqdm
import pickle
import time
from functools import partial

from .nngeometry.object import PMatEKFAC
from .nngeometry.object import lm_vector
from .nngeometry.maths import kronecker

class GDIG:
    """
    G-DIG: Gradient-based DIverse and hiGh-quality data selection for diagnosing model performance.
    """
    
    def __init__(
        self,
        model_name,
        tokenizer_name=None,
        device="cuda",
        lambda_param=0.5,
        batch_query=16,
        ignore_layers=None
    ):
        """
        Initialize the G-DIG algorithm.
        
        Args:
            model_name: Name of the pre-trained model or path to model
            tokenizer_name: Name of the tokenizer (defaults to model_name if None)
            device: Device to use for computation ("cuda", "cuda:0", "cpu", etc.)
            lambda_param: Regularization parameter for the inverse Hessian
            batch_query: Batch size for query data
            ignore_layers: List of layer name patterns to ignore (defaults to bloom layers if None)
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.device = device
        self.lambda_param = lambda_param
        self.batch_query = batch_query
        
        # Default ignore layers for BLOOM models
        self.ignore_layers = ignore_layers if ignore_layers else ['atte', 'lm_head', 'dense_4h_to_h']
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to(self.device)
        
        self.kfac = None
        self.d_ihvp = None

    def to_device(self, kfac, device):
        """Move KFAC matrices to specified device."""
        for key in kfac.data.keys():
            kfac.data[key] = (kfac.data[key][0].to(device), kfac.data[key][1].to(device))
        torch.cuda.empty_cache()
        return kfac

    def to_ekfac_and_device(self, kfac, device):
        """Convert KFAC to EKFAC and move to specified device."""
        for key in kfac.data.keys():
            kfac.data[key] = (kfac.data[key][0].to(device), kfac.data[key][1].to(device))
        
        evecs = dict()
        diags = dict()

        kfac_blocks = kfac.data
        for layer_id, layer in kfac.generator.layer_collection.layers.items():
            a, g = kfac_blocks[layer_id]
            evals_a, evecs_a = torch.linalg.eigh(a)
            evals_g, evecs_g = torch.linalg.eigh(g)
            evecs[layer_id] = (evecs_a, evecs_g)
            diags[layer_id] = kronecker(evals_g.view(-1, 1), evals_a.view(-1, 1))
            del a, g, kfac_blocks[layer_id]
        
        data = (evecs, diags)
        ekfac = PMatEKFAC(generator=kfac.generator, data=data)
        return ekfac

    def load_kfac(self, kfac_path, use_ekfac=False):
        """
        Load pre-computed KFAC matrices.
        
        Args:
            kfac_path: Path to the pre-computed KFAC matrices
            use_ekfac: Whether to convert KFAC to EKFAC
        """
        with open(kfac_path, 'rb') as f:
            self.kfac = pickle.load(f)
            if use_ekfac:
                self.kfac = self.to_ekfac_and_device(self.kfac, self.device)
            else:
                self.kfac = self.to_device(self.kfac, self.device)
        return self

    def prepare_query_data(self, query_dataset, prompt_maker=None, limit_query=-1):
        """
        Prepare the query data for influence computation.
        
        Args:
            query_dataset: Query dataset or path to query data
            prompt_maker: Optional prompt maker for formatting data
            limit_query: Limit the number of query examples (-1 for all)
        """
        # If query_dataset is a string, load the dataset
        if isinstance(query_dataset, str):
            from .dataset.data.json_data import get_json_train_valid_data, generate_and_tokenize_prompt
            from .dataset.prompt_maker.translate_prompt_maker import PromptMaker
            
            class TmpArgs:
                def __init__(self):
                    self.max_length = 256
                    self.use_prompt_loss = False
                    self.prob_data_display = 0.1
                    self.data_path = query_dataset
                    self.valid_data_path = None
                    self.use_large_data = False
                    self.val_set_size = None
                    self.micro_batch_size = 4
                    self.tokenizer = self.tokenizer_name
                    self.seed = 1
            
            tmp_args = TmpArgs()
            prompt_maker = prompt_maker or PromptMaker(args=tmp_args)
            
            query_data, _ = get_json_train_valid_data(
                args=tmp_args,
                data_file=tmp_args.data_path,
                valid_data_file=tmp_args.valid_data_path,
                val_set_size=tmp_args.val_set_size,
                prompt_fn=partial(generate_and_tokenize_prompt, args=tmp_args, verbose=False, tokenizer=self.tokenizer, prompt_maker=prompt_maker),
            )
        else:
            query_data = query_dataset

        if limit_query > 0:
            query_data = torch.utils.data.Subset(query_data, range(limit_query))
        
        queryloader = DataLoader(
            query_data,
            shuffle=False, 
            collate_fn=transformers.DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
            batch_size=self.batch_query, 
            pin_memory=False,
            drop_last=True
        )
        
        # Compute influence vectors for query data
        self.d_ihvp = []
        for q in tqdm(queryloader, desc="Processing query data"):
            self.model.zero_grad()
            inp = q['input_ids'].to(self.device)
            labels = q['labels'].to(self.device)
            
            loss = self.model(input_ids=inp, labels=labels).loss
            loss.backward()

            vec_query = lm_vector.PVector.from_model_grad(self.model, ignore_layers=self.ignore_layers)
            ihvp = self.kfac.inverse(regul=self.lambda_param).mv(vec_query)
            ihvp = lm_vector.PVector(layer_collection=ihvp.layer_collection, 
                                    vector_repr=ihvp.vector_repr, dict_repr=ihvp.dict_repr)
            ihvp.svd()
            self.d_ihvp.append(ihvp)
        
        return self

    def score_data(self, candidate_dataset, prompt_maker=None, limit=-1, start_idx=0, end_idx=None, return_full_score=False):
        """
        Score candidate data based on the influence function.
        
        Args:
            candidate_dataset: Candidate dataset or path to candidate data
            prompt_maker: Optional prompt maker for formatting data
            limit: Limit the number of candidate examples (-1 for all)
            start_idx: Start index for scoring
            end_idx: End index for scoring
            return_full_score: Whether to return all scores for each query
            
        Returns:
            A list of dictionaries containing scores and data text
        """
        if self.kfac is None:
            raise ValueError("KFAC matrices must be loaded first. Use load_kfac method.")
        
        if self.d_ihvp is None:
            raise ValueError("Query data must be prepared first. Use prepare_query_data method.")
        
        # If candidate_dataset is a string, load the dataset
        if isinstance(candidate_dataset, str):
            from .dataset.data.json_data import get_json_train_valid_data, generate_and_tokenize_prompt
            from .dataset.prompt_maker.translate_prompt_maker import PromptMaker
            
            class TmpArgs:
                def __init__(self):
                    self.max_length = 256
                    self.use_prompt_loss = False
                    self.prob_data_display = 0.1
                    self.data_path = candidate_dataset
                    self.valid_data_path = None
                    self.use_large_data = False
                    self.val_set_size = None
                    self.micro_batch_size = 4
                    self.tokenizer = self.tokenizer_name
                    self.seed = 1
            
            tmp_args = TmpArgs()
            prompt_maker = prompt_maker or PromptMaker(args=tmp_args)
            
            candidate_data, _ = get_json_train_valid_data(
                args=tmp_args,
                data_file=tmp_args.data_path,
                valid_data_file=tmp_args.valid_data_path,
                val_set_size=tmp_args.val_set_size,
                prompt_fn=partial(generate_and_tokenize_prompt, args=tmp_args, verbose=False, tokenizer=self.tokenizer, prompt_maker=prompt_maker),
            )
        else:
            candidate_data = candidate_dataset
        
        if limit > 0:
            candidate_data = torch.utils.data.Subset(candidate_data, range(limit))
        else:
            if end_idx is not None:
                ids = list(range(start_idx, end_idx+1))
                candidate_data = torch.utils.data.Subset(candidate_data, ids)
        
        dataloader = DataLoader(
            candidate_data,
            shuffle=False, 
            collate_fn=transformers.DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
            batch_size=1, 
            pin_memory=False,
            drop_last=True
        )
        
        results = []
        for data in tqdm(dataloader, desc="Scoring candidate data"):
            inp = data['input_ids'].to(self.device)
            labels = data['labels'].to(self.device)

            self.model.zero_grad()
            loss = self.model(input_ids=inp, labels=labels).loss
            loss.backward()

            vec_candi = lm_vector.PVector.from_model_grad(self.model, self.ignore_layers)
            vec_candi.svd()
            
            score = 0
            score_list = []

            for ihvp in self.d_ihvp:
                tmp = -(vec_candi.dot_svd(ihvp))
                score = score + tmp
                score_list.append(tmp.item())

            score = score/len(self.d_ihvp)
            text = self.tokenizer.batch_decode(data['input_ids'], skip_special_tokens=True)
            
            if return_full_score:
                results.append({
                    'score': score.item(), 
                    'score_list': score_list, 
                    'loss': loss.item(), 
                    'text': text
                })
            else:
                results.append({
                    'score': score.item(), 
                    'loss': loss.item(), 
                    'text': text
                })
        
        # Sort results by score
        results = sorted(results, key=lambda x: x['score'])
        return results

    def save_results(self, results, output_path):
        """
        Save scoring results to a file.
        
        Args:
            results: Scoring results from score_data method
            output_path: Path to save the results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        return output_path 