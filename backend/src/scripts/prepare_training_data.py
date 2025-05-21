import argparse
import pandas as pd
import json
import os
from typing import Dict, Any

def process_training_dataset(input_file: str, output_file: str) -> None:
    """Process the training dataset to format required for fine-tuning."""
    print(f"Processing dataset from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check required columns
    required_columns = ["Prompt", "Tone", "Channel", "Industry", "Generated_Content"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Transform data into format for fine-tuning
    processed_data = []
    
    for _, row in df.iterrows():
        # Create metadata
        metadata = {
            "tone": row["Tone"],
            "channel": row["Channel"],
            "industry": row["Industry"]
        }
        
        # Create training example
        processed_data.append({
            "content_type": row["Channel"].lower(),
            "prompt": row["Prompt"],
            "content": row["Generated_Content"],
            "metadata": json.dumps(metadata)
        })
    
    # Create DataFrame with processed data
    processed_df = pd.DataFrame(processed_data)
    
    # Save to output file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    print(f"Processed {len(processed_df)} records")

def main():
    """Main function to prepare training data."""
    parser = argparse.ArgumentParser(description="Prepare training data for content generation model")
    parser.add_argument("--input", required=True, help="Path to input CSV file with content data")
    parser.add_argument("--output", required=True, help="Path to output CSV file for training")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    process_training_dataset(args.input, args.output)

if __name__ == "__main__":
    main()