# G-DIG: Gradient-based DIverse and hiGh-quality Data Selection

G-DIG is a tool for diagnosing why models perform poorly on new data, especially on knowledge-based data the model hasn't seen before. It uses gradient information to calculate influence scores that identify problematic or high-quality examples.

## Installation

```bash
pip install gdig
```

Or install from the source:

```bash
git clone https://github.com/yourusername/G-DIG.git
cd G-DIG
pip install -e .
```

## Features

- Calculate influence scores for data examples to identify which ones may cause model performance issues
- Analyze how new, unseen data affects model prediction based on gradient information
- Efficiently select high-quality training data for fine-tuning
- Command-line interface for easy usage

## Quick Start

### Command-line Usage

```bash
# Score data using G-DIG
gdig score -m bigscience/bloom-560m -k path/to/kfac.pkl -d path/to/candidate_data.json -q path/to/query_data.json -o results.json
```

### Python API Usage

```python
from gdig.gdig import GDIG

# Initialize G-DIG
gdig = GDIG(
    model_name="bigscience/bloom-560m",
    device="cuda"
)

# Load pre-computed KFAC matrices
gdig.load_kfac("path/to/kfac.pkl")

# Prepare query data
gdig.prepare_query_data("path/to/query_data.json")

# Score candidate data
results = gdig.score_data("path/to/candidate_data.json")

# Save results
gdig.save_results(results, "results.json")
```

## Documentation

### GDIG Class

The main class for the G-DIG algorithm.

#### Parameters:
- `model_name`: Name of the pre-trained model or path to model
- `tokenizer_name`: Name of the tokenizer (defaults to model_name if None)
- `device`: Device to use for computation ("cuda", "cuda:0", "cpu", etc.)
- `lambda_param`: Regularization parameter for the inverse Hessian
- `batch_query`: Batch size for query data
- `ignore_layers`: List of layer name patterns to ignore

#### Main Methods:
- `load_kfac(kfac_path, use_ekfac=False)`: Load pre-computed KFAC matrices
- `prepare_query_data(query_dataset, prompt_maker=None, limit_query=-1)`: Prepare the query data for influence computation
- `score_data(candidate_dataset, prompt_maker=None, limit=-1, start_idx=0, end_idx=None, return_full_score=False)`: Score candidate data based on the influence function
- `save_results(results, output_path)`: Save scoring results to a file

## Citation

If you use G-DIG in your research, please cite:

```
@article{pan2024g,
  title={G-DIG: Towards Gradient-based DIverse and hiGh-quality Instruction Data Selection for Machine Translation},
  author={Pan, Xingyuan and Huang, Luyang and Kang, Liyan and Liu, Zhicheng and Lu, Yu and Cheng, Shanbo},
  journal={arXiv preprint arXiv:2405.12915},
  year={2024}
}
```

## License

MIT 