import os
import sys
import argparse
from datetime import datetime
from typing import Optional



from prepare_training_data import process_training_dataset
from fine_tune_model import fine_tune_model
from app.core.config import settings

def run_training_pipeline(
    input_data: str,
    output_dir: Optional[str] = None,
    use_lora: bool = True,
    merge_weights: bool = True
) -> None:
    """Run the complete training pipeline."""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up directories
    if output_dir is None:
        output_dir = os.path.join(settings.MODEL_DIR, f"training_run_{timestamp}")
    
    # Create processed data path
    processed_data_path = os.path.join(output_dir, "processed_data.csv")
    
    print("=== Starting Training Pipeline ===")
    print(f"Input data: {input_data}")
    print(f"Output directory: {output_dir}")
    print(f"Using LoRA: {use_lora}")
    print(f"Merge weights: {merge_weights}")
    
    # Step 1: Prepare training data
    print("\n=== Step 1: Preparing Training Data ===")
    process_training_dataset(input_data, processed_data_path)
    
    # Step 2: Fine-tune model
    print("\n=== Step 2: Fine-tuning Model ===")
    fine_tune_model(
        dataset_path=processed_data_path,
        output_dir=output_dir,
        use_lora=use_lora,
        merge_weights=merge_weights
    )
    
    print("\n=== Training Pipeline Completed ===")
    print(f"Final model saved to: {output_dir}")
    if use_lora and merge_weights:
        print(f"Merged model saved to: {os.path.join(output_dir, 'merged_model')}")

def main():
    parser = argparse.ArgumentParser(description="Run the complete training pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV file with content data")
    parser.add_argument("--output", help="Output directory for the trained model")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA fine-tuning")
    parser.add_argument("--no-merge", action="store_true", help="Disable merging LoRA weights")
    
    args = parser.parse_args()
    
    run_training_pipeline(
        input_data=args.input,
        output_dir=args.output,
        use_lora=not args.no_lora,
        merge_weights=not args.no_merge
    )

if __name__ == "__main__":
    main() 