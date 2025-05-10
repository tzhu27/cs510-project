#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from gdig_py.config import GDIGConfig
from gdig_py.pipeline import GDIGPipeline

debug = False
def main():
    parser = argparse.ArgumentParser(
        description="G-DIG: Gradient-based Diverse and high-quality Instruction data selection"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    if not debug:
        try:
            # Load and validate configuration
            config = GDIGConfig.from_yaml(args.config)
            config.validate()
            
            # Run pipeline
            pipeline = GDIGPipeline(config)
            
            pipeline.run()

        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        # Load and validate configuration
        config = GDIGConfig.from_yaml(args.config)
        config.validate()
            
        pipeline = GDIGPipeline(config)
        pipeline.debug()

if __name__ == "__main__":
    main() 