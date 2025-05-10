# G-DIG: Gradient-based Diverse and high-quality Instruction data selection

G-DIG is a Python package for selecting high-quality instruction data for machine translation using gradient-based methods. It implements a novel approach that combines influence functions with clustering to select diverse and high-quality training examples.

## Installation

```bash
pip install gdig-py
```

## Quick Start

1. Create a configuration file `config.yaml`:

```yaml
# Data paths
seed_set_path:         data/seed.json
candidate_pool_path:   data/candidate.json
model_checkpoint:      bigscience/bloom-560m
tokenizer_path:        bigscience/bloom-560m  # Can be same as model_checkpoint but on huggingface

# Hessian computation parameters
hessian_device:        cuda:0
num_gpus:             8
hessian_trials:       1
hessian_output:       hessian.pkl

# Influence score parameters
influence_batch_size:  2
influence_lambda:      0.5
full_score:           false
use_ekfac:            false
start_index:          null  # Optional
end_index:            null  # Optional
layer_type:           b

# Clustering parameters
num_clusters:         50
cluster_metric:       cosine

# Model-specific settings
model_type:           bloom  # One of: bloom, llama, baichuan
max_length:           256
micro_batch_size:     4
use_prompt_loss:      false

# General settings
output_dir:           results/gdig_run1
log_level:            INFO
visualize:            false
```

2. Run the pipeline:

```bash
python -m gdig_py.main --config config.yaml
```

## Output Structure

The pipeline generates the following outputs in the specified `output_dir`:

- `selected.json` — Final list of selected examples
- `influence.csv` — Per-candidate influence statistics (mean, min, max)
- `clusters/` — Cluster assignments and diagnostics
- `logs/` — Progress logs and performance summaries

## Configuration Options

### Data Paths
- `seed_set_path`: Path to seed examples JSON file
- `candidate_pool_path`: Path to candidate pool JSON file
- `model_checkpoint`: Path to pre-trained model checkpoint
- `tokenizer_path`: Path to tokenizer (can be same as model_checkpoint)

### Hessian Computation Parameters
- `hessian_device`: GPU device for Hessian computation (e.g., "cuda:0")
- `num_gpus`: Number of GPUs to use for parallel computation
- `hessian_trials`: Number of trials for Hessian computation
- `hessian_output`: Output filename for Hessian matrix

### Influence Score Parameters
- `influence_batch_size`: Batch size for influence computation
- `influence_lambda`: Lambda parameter for Hessian inverse
- `full_score`: Whether to compute full score statistics
- `use_ekfac`: Whether to use EKFAC instead of KFAC
- `start_index`: Optional start index for processing subset
- `end_index`: Optional end index for processing subset
- `layer_type`: Layer type for computation (default: "b")

### Clustering Parameters
- `num_clusters`: Number of clusters for diversity
- `cluster_metric`: Distance metric for clustering ('cosine' or 'euclidean')

### Model-specific Settings
- `model_type`: Type of model ('bloom', 'llama', or 'baichuan')
- `max_length`: Maximum sequence length
- `micro_batch_size`: Micro batch size for processing
- `use_prompt_loss`: Whether to use prompt loss

### General Settings
- `output_dir`: Directory for saving results
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `visualize`: Whether to generate visualization plots

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{pan2024g,
  title={G-DIG: Towards Gradient-based DIverse and hiGh-quality Instruction Data Selection for Machine Translation},
  author={Pan, Xingyuan and Huang, Luyang and Kang, Liyan and Liu, Zhicheng and Lu, Yu and Cheng, Shanbo},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={15395--15406},
  year={2024}
}
``` 