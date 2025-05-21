import os
import json
import torch
from typing import Dict, Optional, Any


from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import pandas as pd
from sqlalchemy.orm import Session
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)

from app.models.content import Content
from app.core.config import settings

class ContentGenerationService:
    """Service for generating AI content based on tone and prompt."""
    
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    MODEL_VERSION = "v1.0.0"
    
    # Generation parameters
    DEFAULT_MAX_LENGTH = 1024
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    DEFAULT_TOP_K = 50
    
    # Available tones based on UI
    TONES = ["Professional", "Casual", "Formal", "Friendly", "Authoritative"]
    
    def __init__(self):
        # Get Hugging Face token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set. Please set it with your Hugging Face token.")
        
        # Initialize model and tokenizer
        self.model_name = "mistralai/Mistral-7B-v0.1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=self.hf_token
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            token=self.hf_token,
            device_map="auto",
            load_in_8bit=True  # Enable 8-bit quantization for memory efficiency
        )
        
        print("Model loaded successfully!")
        
        self.model.to(self.device)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_content(
        self, 
        prompt: str,
        tone: str = "Professional",
        industry: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> str:
        """Generate content based on prompt and tone."""
        # Validate tone
        if tone not in self.TONES:
            tone = "Professional"  # Default to professional if invalid tone
        
        # Construct prompt
        formatted_prompt = self._construct_prompt(prompt, tone, industry, channel)
        
        # Set generation parameters
        generation_params = {
            "max_length": self.DEFAULT_MAX_LENGTH,
            "temperature": self.DEFAULT_TEMPERATURE,
            "top_p": self.DEFAULT_TOP_P,
            "top_k": self.DEFAULT_TOP_K,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Encode prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate content
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                **generation_params
            )
        
        # Decode and clean output
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        result = generated_text[len(formatted_prompt):].strip()
        
        return result
    
    def save_generated_content(
        self,
        db: Session,
        content: str,
        prompt: str,
        tone: str,
        user_id: int,
        channel: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> Content:
        """Save generated content to database."""
        # Create content object
        db_content = Content(
            title=prompt[:100] if prompt else f"Generated content with {tone} tone",
            body=content,
            type=channel.lower() if channel else "general",
            status="draft",
            ai_generated=True,
            generation_params={
                "prompt": prompt,
                "tone": tone,
                "industry": industry,
                "channel": channel
            },
            model_version=self.MODEL_VERSION,
            created_by_id=user_id,
            target_audience=None,  # Not provided in the UI
        )
        
        db.add(db_content)
        db.commit()
        db.refresh(db_content)
        
        return db_content
    
    def merge_lora_weights(self, output_dir: str) -> None:
        """Merge LoRA weights with the base model for inference."""
        if not isinstance(self.model, PeftModel):
            print("Model is not a PEFT model, skipping merge")
            return
            
        print("Merging LoRA weights with base model...")
        # Merge weights
        self.model = self.model.merge_and_unload()
        
        # Save merged model
        merged_model_path = os.path.join(output_dir, "merged_model")
        os.makedirs(merged_model_path, exist_ok=True)
        
        self.model.save_pretrained(merged_model_path)
        self.tokenizer.save_pretrained(merged_model_path)
        
        print(f"Merged model saved to {merged_model_path}")
        
        # Reload the merged model
        self.model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.to(self.device)
        print("Merged model loaded successfully")
    
    def fine_tune_model(
        self, 
        dataset_path: str, 
        output_dir: Optional[str] = None,
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        merge_weights: bool = True
    ) -> None:
        """Fine-tune the model using the provided dataset with LoRA."""
        # Set output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(settings.MODEL_DIR, "mistral-7b-marketing-lora" if use_lora else "mistral-7b-marketing")
        
        # Ensure model directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare dataset
        train_dataset = self._prepare_dataset(dataset_path)
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        if use_lora:
            # Default LoRA config if none provided
            if lora_config is None:
                lora_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": TaskType.CAUSAL_LM,
                    "target_modules": ["q_proj", "v_proj"],
                    "modules_to_save": None,
                }
            
            # Create LoRA config
            peft_config = LoraConfig(**lora_config)
            
            # Get PEFT model
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            save_steps=1000,
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=self.device == "cuda",
            logging_dir=os.path.join(output_dir, "logs"),
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=10,
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're using causal language modeling, not masked
        )
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        if use_lora:
            # Save LoRA adapter
            self.model.save_pretrained(output_dir)
            
            # Merge weights if requested
            if merge_weights:
                self.merge_lora_weights(output_dir)
        else:
            # Save full model
            trainer.save_model(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        
        # Reload the fine-tuned model
        if use_lora and not merge_weights:
            # Load base model and LoRA adapter
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                load_in_8bit=True
            )
            self.model = get_peft_model(self.model, peft_config)
        else:
            # Load full model (either merged or full fine-tuned)
            model_path = os.path.join(output_dir, "merged_model") if use_lora and merge_weights else output_dir
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
        self.model.to(self.device)
    
    def _prepare_dataset(self, dataset_path: str) -> Dataset:
        """Prepare dataset for fine-tuning."""
        # Load CSV dataset
        df = pd.read_csv(dataset_path)
        
        # Check required columns
        if "prompt" not in df.columns or "content" not in df.columns:
            raise ValueError("Dataset missing required columns: prompt, content")
        
        # Format for training
        texts = []
        for _, row in df.iterrows():
            # Get prompt and generated content
            prompt = row["prompt"]
            content = row["content"]
            
            # Get metadata if available
            metadata = {}
            if "metadata" in df.columns and pd.notna(row["metadata"]):
                try:
                    metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
                except json.JSONDecodeError:
                    pass
            
            # Create prompt with tone and other metadata
            formatted_prompt = self._construct_training_prompt(
                prompt, 
                metadata.get("tone", "Professional"),
                metadata.get("industry"),
                metadata.get("channel")
            )
            
            # Combine prompt and content
            full_text = f"{formatted_prompt}\n\n{content}"
            texts.append(full_text)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def _construct_training_prompt(
        self, 
        prompt: str, 
        tone: str = "Professional",
        industry: Optional[str] = None,
        channel: Optional[str] = None
    ) -> str:
        """Construct prompt for training data with the same format as inference."""
        return self._construct_prompt(prompt, tone, industry, channel)
    
    def _construct_prompt(
        self, 
        prompt: str, 
        tone: str = "Professional",
        industry: Optional[str] = None,
        channel: Optional[str] = None
    ) -> str:
        """Construct prompt for generation."""
        formatted_prompt = f"Generate content with a {tone} tone"
        
        # Add channel if provided
        if channel:
            formatted_prompt += f" for {channel}"
        
        # Add industry if provided
        if industry:
            formatted_prompt += f" in the {industry} industry"
        
        # Add the actual prompt
        formatted_prompt += f":\n\n{prompt}\n\nGenerated content:"
        
        return formatted_prompt