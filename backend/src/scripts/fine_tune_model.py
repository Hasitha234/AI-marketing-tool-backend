import sys
import os
import argparse
import pandas as pd
import json
from typing import Optional, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




from app.services.content_generation import ContentGenerationService
from app.core.config import settings

def get_default_dataset_path() -> str:
    """Get the default path to the content dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "tests", "content_dataset.csv")

def get_lora_config() -> Dict[str, Any]:
    """Get LoRA configuration parameters."""
    return {
        "r": 16,  # LoRA attention dimension
        "lora_alpha": 32,  # LoRA alpha parameter
        "lora_dropout": 0.05,  # LoRA dropout
        "bias": "none",  # LoRA bias type
        "task_type": "CAUSAL_LM",  # Task type for causal language modeling
        "target_modules": ["q_proj", "v_proj"],  # Target attention modules
        "modules_to_save": None,  # No additional modules to save
    }

def process_marketing_dataset(dataset_path: str, output_path: str) -> str:
    """Process the marketing dataset to the format needed for fine-tuning."""
    # Read CSV dataset
    df = pd.read_csv(dataset_path)
    
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
            "prompt": row["Prompt"],
            "content": row["Generated_Content"],
            "metadata": json.dumps(metadata)
        })
    
    # Create DataFrame with processed data
    processed_df = pd.DataFrame(processed_data)
    
    # Save to temporary file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_df.to_csv(output_path, index=False)
    
    return output_path

def fine_tune_model(
    dataset_path: str = None,
    output_dir: str = None,
    use_lora: bool = True,
    lora_config: Optional[Dict[str, Any]] = None
) -> None:
    """Fine-tune the content generation model using LoRA."""
    # Use default dataset if none provided
    if dataset_path is None:
        dataset_path = get_default_dataset_path()
        print(f"Using default dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Process the dataset to the required format
    temp_processed_path = os.path.join(os.path.dirname(dataset_path), "processed_" + os.path.basename(dataset_path))
    processed_path = process_marketing_dataset(dataset_path, temp_processed_path)
    
    print(f"Processed dataset saved to {processed_path}")
    
    # Initialize content generation service
    content_service = ContentGenerationService()
    
    # Set output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(settings.MODEL_DIR, "mistral-7b-marketing-lora" if use_lora else "mistral-7b-marketing")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    # Get LoRA configuration if using LoRA
    if use_lora and lora_config is None:
        lora_config = get_lora_config()
    
    print(f"Starting model fine-tuning with dataset: {processed_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using LoRA: {use_lora}")
    if use_lora:
        print(f"LoRA configuration: {json.dumps(lora_config, indent=2)}")
    
    # Fine-tune model
    try:
        content_service.fine_tune_model(
            dataset_path=processed_path,
            output_dir=output_dir,
            use_lora=use_lora,
            lora_config=lora_config
        )
        print("Model fine-tuning completed successfully!")
    except Exception as e:
        print(f"Error during model fine-tuning: {str(e)}")
    
    # Clean up temporary file
    if os.path.exists(processed_path):
        os.remove(processed_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the content generation model")
    parser.add_argument("--dataset", help="Path to the CSV dataset file (defaults to tests/content_dataset.csv)")
    parser.add_argument("--output", help="Output directory for the fine-tuned model")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA fine-tuning")
    
    args = parser.parse_args()
    
    fine_tune_model(
        dataset_path=args.dataset,
        output_dir=args.output,
        use_lora=not args.no_lora
    )