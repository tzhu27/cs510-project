"""
Command-line interface for the G-DIG package.
"""
import argparse
import os
import json
from .gdig import GDIG

def main():
    parser = argparse.ArgumentParser(
                    prog='gdig',
                    description='G-DIG: Gradient-based DIverse and hiGh-quality data selection',
                    epilog='For more information, visit: https://github.com/yourusername/G-DIG')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Command: score
    score_parser = subparsers.add_parser('score', help='Score data using G-DIG algorithm')
    score_parser.add_argument('-m', '--model', required=True, help='Model name or path')
    score_parser.add_argument('-t', '--tokenizer', help='Tokenizer name or path (defaults to model if not specified)')
    score_parser.add_argument('-k', '--kfac', required=True, help='Path to pre-computed KFAC matrices')
    score_parser.add_argument('-d', '--data', required=True, help='Path to candidate data')
    score_parser.add_argument('-q', '--query', required=True, help='Path to query data')
    score_parser.add_argument('-o', '--output', default='gdig_results.json', help='Output file path')
    score_parser.add_argument('-v', '--device', default='cuda', help='Device to use (cuda, cuda:0, cpu, etc.)')
    score_parser.add_argument('-l', '--limit', type=int, default=-1, help='Limit the number of candidate examples (-1 for all)')
    score_parser.add_argument('-lq', '--limit_query', type=int, default=-1, help='Limit the number of query examples (-1 for all)')
    score_parser.add_argument('-lmd', '--lambda', dest='lambda_param', type=float, default=0.5, help='Regularization parameter for inverse Hessian')
    score_parser.add_argument('-bq', '--batch_query', type=int, default=16, help='Batch size for query data')
    score_parser.add_argument('--full-score', action='store_true', help='Return detailed scores for each query')
    score_parser.add_argument('--ekfac', action='store_true', help='Use EKFAC instead of KFAC')
    score_parser.add_argument('--start', type=int, default=0, help='Start index for scoring')
    score_parser.add_argument('--end', type=int, default=None, help='End index for scoring')
    
    args = parser.parse_args()
    
    if args.command == 'score':
        # Initialize G-DIG
        gdig = GDIG(
            model_name=args.model,
            tokenizer_name=args.tokenizer,
            device=args.device,
            lambda_param=args.lambda_param,
            batch_query=args.batch_query
        )
        
        # Load KFAC matrices
        gdig.load_kfac(args.kfac, use_ekfac=args.ekfac)
        
        # Prepare query data
        gdig.prepare_query_data(args.query, limit_query=args.limit_query)
        
        # Score candidate data
        results = gdig.score_data(
            args.data,
            limit=args.limit,
            start_idx=args.start,
            end_idx=args.end,
            return_full_score=args.full_score
        )
        
        # Save results
        output_path = gdig.save_results(results, args.output)
        print(f"Results saved to {output_path}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 