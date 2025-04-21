"""
Basic usage example for the G-DIG package.
"""
import json
from gdig.gdig import GDIG

def main():
    # Initialize G-DIG
    gdig = GDIG(
        model_name="bigscience/bloom-560m",  # Example model
        device="cuda"  # Use "cpu" if no GPU is available
    )
    
    # Load pre-computed KFAC matrices
    gdig.load_kfac("path/to/kfac.pkl")
    
    # Prepare query data
    gdig.prepare_query_data("path/to/query_data.json", limit_query=10)
    
    # Score candidate data
    results = gdig.score_data("path/to/candidate_data.json", limit=50)
    
    # Save results
    gdig.save_results(results, "gdig_results.json")
    
    # Print top 3 results
    print("Top 3 examples with lowest influence score (indicating potential issues):")
    for i, result in enumerate(results[:3]):
        print(f"Example {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Text: {result['text'][0][:100]}..." if len(result['text'][0]) > 100 else result['text'][0])
        print()

if __name__ == "__main__":
    main() 