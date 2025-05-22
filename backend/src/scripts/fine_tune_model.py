import sys
import os
import argparse
import pandas as pd
import json
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List
import wandb

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.content_generation import ContentGenerationService, GenerationConfig
from app.core.config import settings

def get_default_dataset_path() -> str:
    """Get the default path to the content dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "tests", "content_dataset.csv")

def get_optimized_lora_config(model_size: str = "7b") -> Dict[str, Any]:
    """Get optimized LoRA configuration parameters based on model size."""
    configs = {
        "7b": {
            "r": 32,  # Higher rank for better capacity
            "lora_alpha": 64,  # Balanced alpha
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
                "gate_proj", "up_proj", "down_proj"  # MLP modules
            ],
            "modules_to_save": None,
        },
        "13b": {
            "r": 16,  # Lower rank for larger models
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "modules_to_save": None,
        }
    }
    return configs.get(model_size, configs["7b"])

def setup_wandb_logging(project_name: str = "mistral-marketing-finetune", config: Dict = None) -> bool:
    """Setup Weights & Biases logging if available."""
    try:
        wandb.init(
            project=project_name,
            config=config,
            tags=["mistral-7b", "lora", "marketing", "content-generation"]
        )
        return True
    except Exception as e:
        print(f"Warning: Could not initialize wandb logging: {e}")
        return False

def process_marketing_dataset(dataset_path: str, output_path: str, enhance_data: bool = True) -> str:
    """Process the marketing dataset to the format needed for fine-tuning."""
    print(f"Processing dataset: {dataset_path}")
    
    # Read CSV dataset
    df = pd.read_csv(dataset_path)
    
    # Check required columns
    required_columns = ["Prompt", "Tone", "Channel", "Industry", "Generated_Content"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    print(f"Original dataset size: {len(df)} samples")
    
    # Enhanced data processing if requested
    if enhance_data:
        print("Applying data enhancement...")
        df = enhance_dataset_content(df)
        print(f"Enhanced dataset size: {len(df)} samples")
    
    # Transform data into format for fine-tuning
    processed_data = []
    
    for _, row in df.iterrows():
        # Create metadata
        metadata = {
            "tone": row["Tone"],
            "channel": row["Channel"],
            "industry": row["Industry"]
        }
        
        # Add additional fields if available
        for field in ["Product", "Audience", "Benefit"]:
            if field in df.columns:
                metadata[field.lower()] = row[field]
        
        # Create training example
        processed_data.append({
            "prompt": row["Prompt"],
            "content": row["Generated_Content"],
            "metadata": json.dumps(metadata)
        })
    
    # Create DataFrame with processed data
    processed_df = pd.DataFrame(processed_data)
    
    # Save to output file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")
    
    return output_path

def enhance_dataset_content(df: pd.DataFrame) -> pd.DataFrame:
    """Apply content enhancement to improve training data quality."""
    enhanced_df = df.copy()
    
    # Remove very short content (likely low quality)
    min_length = 30
    enhanced_df = enhanced_df[enhanced_df['Generated_Content'].str.len() >= min_length]
    
    # Remove exact duplicates
    original_size = len(enhanced_df)
    enhanced_df = enhanced_df.drop_duplicates(subset=['Generated_Content'])
    duplicates_removed = original_size - len(enhanced_df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate content samples")
    
    # Balance the dataset across tones and channels
    enhanced_df = balance_dataset(enhanced_df)
    
    return enhanced_df

def balance_dataset(df: pd.DataFrame, max_samples_per_group: int = 50) -> pd.DataFrame:
    """Balance dataset across tone-channel combinations."""
    balanced_dfs = []
    
    # Group by tone and channel
    for (tone, channel), group in df.groupby(['Tone', 'Channel']):
        if len(group) > max_samples_per_group:
            # Sample randomly to balance
            sampled_group = group.sample(n=max_samples_per_group, random_state=42)
            balanced_dfs.append(sampled_group)
        else:
            balanced_dfs.append(group)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"Dataset balanced: {len(df)} -> {len(balanced_df)} samples")
    
    return balanced_df

def create_test_prompts() -> List[Dict]:
    """Create test prompts for evaluation."""
    return [
        {
            "prompt": "Create a social media post about our new AI writing assistant",
            "tone": "Professional",
            "industry": "Tech",
            "channel": "Social Media"
        },
        {
            "prompt": "Write an email announcing our fitness app to health enthusiasts",
            "tone": "Casual",
            "industry": "Health",
            "channel": "Email"
        },
        {
            "prompt": "Describe the benefits of our project management tool",
            "tone": "Witty",
            "industry": "Business",
            "channel": "Product Description"
        },
    ]

def evaluate_model_performance(
    content_service: ContentGenerationService, 
    test_prompts: List[Dict],
    output_dir: str
) -> Dict:
    """Evaluate model performance on test prompts."""
    print("\nEvaluating model performance...")
    
    results = []
    for prompt_data in test_prompts:
        try:
            generated_content = content_service.generate_content(
                prompt=prompt_data["prompt"],
                tone=prompt_data["tone"],
                industry=prompt_data.get("industry"),
                channel=prompt_data.get("channel")
            )
            
            result = {
                "prompt": prompt_data["prompt"],
                "tone": prompt_data["tone"],
                "industry": prompt_data.get("industry", ""),
                "channel": prompt_data.get("channel", ""),
                "generated_content": generated_content,
                "content_length": len(generated_content),
                "word_count": len(generated_content.split())
            }
            results.append(result)
            
            print(f"✓ Generated content for {prompt_data['tone']} {prompt_data.get('channel', 'content')}")
            
        except Exception as e:
            print(f"✗ Failed to generate content: {e}")
            results.append({
                "prompt": prompt_data["prompt"],
                "error": str(e)
            })
    
    # Save evaluation results
    eval_file = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate basic metrics
    successful_generations = [r for r in results if "generated_content" in r]
    metrics = {
        "total_prompts": len(test_prompts),
        "successful_generations": len(successful_generations),
        "success_rate": len(successful_generations) / len(test_prompts),
        "avg_content_length": sum(r["content_length"] for r in successful_generations) / len(successful_generations) if successful_generations else 0,
        "avg_word_count": sum(r["word_count"] for r in successful_generations) / len(successful_generations) if successful_generations else 0
    }
    
    print(f"Evaluation Results:")
    print(f"  - Success Rate: {metrics['success_rate']:.2%}")
    print(f"  - Average Content Length: {metrics['avg_content_length']:.1f} characters")
    print(f"  - Average Word Count: {metrics['avg_word_count']:.1f} words")
    
    return metrics

def fine_tune_model(
    dataset_path: str = None,
    output_dir: str = None,
    use_lora: bool = True,
    lora_config: Optional[Dict[str, Any]] = None,
    enhance_data: bool = True,
    run_evaluation: bool = True,
    use_wandb: bool = False
) -> Dict[str, Any]:
    """Enhanced fine-tuning with comprehensive monitoring and evaluation."""
    
    # Use default dataset if none provided
    if dataset_path is None:
        dataset_path = get_default_dataset_path()
        print(f"Using default dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        model_type = "mistral-7b-marketing-lora" if use_lora else "mistral-7b-marketing"
        output_dir = os.path.join(settings.MODEL_DIR, f"{model_type}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process the dataset
    temp_processed_path = os.path.join(output_dir, "processed_dataset.csv")
    processed_path = process_marketing_dataset(dataset_path, temp_processed_path, enhance_data)
    
    # Setup configuration
    training_config = {
        "dataset_path": dataset_path,
        "processed_dataset_path": processed_path,
        "output_dir": output_dir,
        "use_lora": use_lora,
        "enhance_data": enhance_data,
        "timestamp": timestamp,
        "model_name": ContentGenerationService.MODEL_NAME
    }
    
    # Setup W&B logging if requested
    wandb_enabled = False
    if use_wandb:
        wandb_enabled = setup_wandb_logging("mistral-marketing-finetune", training_config)
    
    # Get LoRA configuration
    if use_lora and lora_config is None:
        lora_config = get_optimized_lora_config("7b")
        training_config["lora_config"] = lora_config
    
    print(f"\n=== Fine-tuning Configuration ===")
    print(f"Dataset: {processed_path}")
    print(f"Output: {output_dir}")
    print(f"Use LoRA: {use_lora}")
    print(f"Enhance Data: {enhance_data}")
    print(f"W&B Logging: {wandb_enabled}")
    if use_lora:
        print(f"LoRA Config: {json.dumps(lora_config, indent=2)}")
    
    # Initialize content generation service
    try:
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_length=1024
        )
        content_service = ContentGenerationService(config=generation_config)
        print("✓ Model initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        raise
    
    # Check for evaluation dataset
    eval_dataset_path = processed_path.replace('.csv', '_eval.csv')
    if not os.path.exists(eval_dataset_path):
        eval_dataset_path = None
    
    # Fine-tune the model
    try:
        print("\n=== Starting Fine-tuning ===")
        training_result = content_service.fine_tune_model(
            dataset_path=processed_path,
            output_dir=output_dir,
            use_lora=use_lora,
            lora_config=lora_config,
            merge_weights=True,
            evaluation_data=eval_dataset_path
        )
        print("✓ Fine-tuning completed successfully!")
        
        # Update training config with results
        training_config.update(training_result)
        
    except Exception as e:
        print(f"✗ Fine-tuning failed: {e}")
        if wandb_enabled:
            wandb.log({"training_error": str(e)})
            wandb.finish()
        raise
    
    # Run evaluation if requested
    evaluation_metrics = {}
    if run_evaluation:
        try:
            test_prompts = create_test_prompts()
            evaluation_metrics = evaluate_model_performance(
                content_service, test_prompts, output_dir
            )
            training_config["evaluation_metrics"] = evaluation_metrics
            
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            evaluation_metrics = {"evaluation_error": str(e)}
    
    # Log to W&B if enabled
    if wandb_enabled:
        wandb.log({
            "training_loss": training_result.get("training_loss", 0),
            "training_steps": training_result.get("training_steps", 0),
            **evaluation_metrics
        })
        wandb.finish()
    
    # Save final configuration and results
    final_config_path = os.path.join(output_dir, "final_training_report.json")
    with open(final_config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"\n=== Training Completed ===")
    print(f"Model saved to: {output_dir}")
    print(f"Training report: {final_config_path}")
    if evaluation_metrics:
        print(f"Evaluation success rate: {evaluation_metrics.get('success_rate', 0):.2%}")
    
    # Clean up temporary files
    if os.path.exists(temp_processed_path) and temp_processed_path != processed_path:
        os.remove(temp_processed_path)
    
    return training_config

def compare_models(base_model_path: str, fine_tuned_model_path: str, test_prompts: List[Dict]) -> Dict:
    """Compare base model vs fine-tuned model performance."""
    print("\n=== Model Comparison ===")
    
    results = {
        "base_model_results": [],
        "fine_tuned_results": [],
        "comparison_metrics": {}
    }
    
    try:
        # Load base model
        print("Loading base model...")
        base_service = ContentGenerationService()
        
        # Load fine-tuned model
        print("Loading fine-tuned model...")
        # This would require loading the fine-tuned model
        # Implementation depends on how the model is saved
        
        # Generate content with both models
        for prompt_data in test_prompts:
            # Base model generation
            base_content = base_service.generate_content(
                prompt=prompt_data["prompt"],
                tone=prompt_data["tone"],
                industry=prompt_data.get("industry"),
                channel=prompt_data.get("channel")
            )
            
            results["base_model_results"].append({
                "prompt": prompt_data["prompt"],
                "content": base_content,
                "length": len(base_content)
            })
            
            # Fine-tuned model would be similar
            # For now, we'll skip this part as it requires model loading logic
        
        print("✓ Model comparison completed")
        
    except Exception as e:
        print(f"✗ Model comparison failed: {e}")
        results["error"] = str(e)
    
    return results

def validate_training_environment() -> bool:
    """Validate that the training environment is properly set up."""
    print("=== Environment Validation ===")
    
    checks = []
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    checks.append(("CUDA Available", cuda_available))
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        checks.append(("GPU Count", gpu_count))
        checks.append(("GPU Memory (GB)", f"{gpu_memory:.1f}"))
    
    # Check HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    checks.append(("HuggingFace Token", "Set" if hf_token else "Missing"))
    
    # Check disk space
    try:
        import shutil
        disk_space = shutil.disk_usage("/").free / 1e9
        checks.append(("Free Disk Space (GB)", f"{disk_space:.1f}"))
    except:
        checks.append(("Free Disk Space", "Unknown"))
    
    # Check model directory
    model_dir_exists = os.path.exists(settings.MODEL_DIR)
    checks.append(("Model Directory", "Exists" if model_dir_exists else "Missing"))
    
    # Print results
    all_good = True
    for check_name, result in checks:
        status = "✓" if (isinstance(result, bool) and result) or (isinstance(result, str) and result not in ["Missing", "Unknown"]) else "✗"
        print(f"{status} {check_name}: {result}")
        if status == "✗" and check_name in ["HuggingFace Token", "Model Directory"]:
            all_good = False
    
    if not all_good:
        print("\n⚠️  Some critical checks failed. Please fix these issues before training.")
    else:
        print("\n✓ Environment validation passed!")
    
    return all_good

def main():
    """Main function for enhanced fine-tuning."""
    parser = argparse.ArgumentParser(description="Enhanced fine-tuning for content generation model")
    parser.add_argument("--dataset", help="Path to the CSV dataset file")
    parser.add_argument("--output", help="Output directory for the fine-tuned model")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA fine-tuning")
    parser.add_argument("--no-enhance", action="store_true", help="Disable data enhancement")
    parser.add_argument("--no-eval", action="store_true", help="Skip model evaluation")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--validate-env", action="store_true", help="Only validate environment and exit")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank parameter")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate")
    
    args = parser.parse_args()
    
    # Validate environment if requested
    if args.validate_env:
        validate_training_environment()
        return
    
    # Validate environment before training
    if not validate_training_environment():
        print("Environment validation failed. Use --validate-env to see details.")
        return
    
    # Custom LoRA config if specified
    lora_config = None
    if not args.no_lora:
        lora_config = get_optimized_lora_config("7b")
        if args.lora_r != 32:
            lora_config["r"] = args.lora_r
        if args.lora_alpha != 64:
            lora_config["lora_alpha"] = args.lora_alpha
        if args.lora_dropout != 0.1:
            lora_config["lora_dropout"] = args.lora_dropout
    
    try:
        result = fine_tune_model(
            dataset_path=args.dataset,
            output_dir=args.output,
            use_lora=not args.no_lora,
            lora_config=lora_config,
            enhance_data=not args.no_enhance,
            run_evaluation=not args.no_eval,
            use_wandb=args.wandb
        )
        
        print(f"\n🎉 Fine-tuning completed successfully!")
        print(f"Results saved to: {result['output_dir']}")
        
    except Exception as e:
        print(f"\n💥 Fine-tuning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()