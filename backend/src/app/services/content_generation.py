import os
import json
import torch
import random
import numpy as np
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
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

# Evaluation imports
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Warning: Rouge and BertScore not available. Install with: pip install rouge-score bert-score")

from app.models.content import Content
from app.core.config import settings

@dataclass
class GenerationConfig:
    """Configuration for content generation"""
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0

class ModelEvaluator:
    """Comprehensive model evaluation system"""
    
    def __init__(self):
        if EVALUATION_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def evaluate_generation_quality(self, model, test_prompts: List[Dict], reference_outputs: List[str]) -> Dict:
        """Evaluate model performance on test set"""
        if not EVALUATION_AVAILABLE:
            print("Evaluation libraries not available. Skipping detailed evaluation.")
            return {"message": "Evaluation libraries not installed"}
        
        results = {
            "rouge_scores": [],
            "bert_scores": [],
            "length_stats": [],
            "tone_consistency": [],
        }
        
        generated_outputs = []
        for prompt_data in test_prompts:
            generated = model.generate_content(
                prompt=prompt_data["prompt"],
                tone=prompt_data.get("tone", "Professional"),
                industry=prompt_data.get("industry"),
                channel=prompt_data.get("channel")
            )
            generated_outputs.append(generated)
        
        # Calculate ROUGE scores
        for gen, ref in zip(generated_outputs, reference_outputs):
            scores = self.rouge_scorer.score(ref, gen)
            results["rouge_scores"].append({
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
            })
        
        # Calculate BERT scores for semantic similarity
        P, R, F1 = bert_score(generated_outputs, reference_outputs, lang="en", verbose=False)
        results["bert_scores"] = F1.tolist()
        
        # Length analysis
        for gen in generated_outputs:
            results["length_stats"].append(len(gen.split()))
        
        # Tone consistency check
        for i, (gen, prompt_data) in enumerate(zip(generated_outputs, test_prompts)):
            tone = prompt_data.get("tone", "Professional")
            consistency = self._check_tone_consistency(gen, tone)
            results["tone_consistency"].append(consistency)
        
        return self._aggregate_results(results)
    
    def _check_tone_consistency(self, text: str, expected_tone: str) -> float:
        """Check tone consistency based on keywords and patterns"""
        tone_indicators = {
            "Professional": {
                "keywords": ["solution", "deliver", "expertise", "comprehensive", "optimize", "professional", "enterprise"],
                "patterns": ["we are pleased", "introducing", "designed for", "industry-leading"]
            },
            "Casual": {
                "keywords": ["hey", "check out", "awesome", "cool", "great", "perfect", "love"],
                "patterns": ["!", "🎉", "👋", "you'll love", "it's perfect"]
            },
            "Witty": {
                "keywords": ["shake things up", "plot twist", "finally", "who knew", "goodbye boring"],
                "patterns": ["🚀", "⚡", "!", "save you from", "make things fun"]
            }
        }
        
        if expected_tone not in tone_indicators:
            return 0.0
        
        indicators = tone_indicators[expected_tone]
        text_lower = text.lower()
        
        keyword_matches = sum(1 for keyword in indicators["keywords"] if keyword in text_lower)
        pattern_matches = sum(1 for pattern in indicators["patterns"] if pattern.lower() in text_lower)
        
        total_indicators = len(indicators["keywords"]) + len(indicators["patterns"])
        total_matches = keyword_matches + pattern_matches
        
        return total_matches / total_indicators if total_indicators > 0 else 0.0
    
    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate evaluation results"""
        return {
            "average_rouge1": np.mean([r["rouge1"] for r in results["rouge_scores"]]),
            "average_rouge2": np.mean([r["rouge2"] for r in results["rouge_scores"]]),
            "average_rougeL": np.mean([r["rougeL"] for r in results["rouge_scores"]]),
            "average_bert_score": np.mean(results["bert_scores"]),
            "average_length": np.mean(results["length_stats"]),
            "length_std": np.std(results["length_stats"]),
            "tone_consistency": np.mean(results["tone_consistency"]),
            "total_samples": len(results["rouge_scores"])
        }

class ContentGenerationService:
    """Enhanced service for generating AI content based on tone and prompt."""
    
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    MODEL_VERSION = "v1.1.0"  # Updated version
    
    # Available tones based on UI
    TONES = ["Professional", "Casual", "Witty"]
    
    # Enhanced prompt templates
    PROMPT_TEMPLATES = {
        "Professional": [
            """As a professional marketing expert in the {industry} industry, create {channel} content that:
- Maintains a professional, authoritative tone
- Highlights key benefits and value propositions
- Uses industry-appropriate terminology
- Builds trust and credibility

Request: {prompt}

Professional {channel} content:""",
            """Create compelling {channel} content for {industry} professionals that demonstrates expertise and builds confidence in our solution.

Task: {prompt}

Content:"""
        ],
        "Casual": [
            """Write friendly, approachable {channel} content for the {industry} space that:
- Uses conversational, everyday language
- Feels personal and relatable
- Includes appropriate emojis and casual expressions
- Makes complex topics accessible

Request: {prompt}

Casual {channel} content:""",
            """Hey there! Help create some casual {channel} content that {industry} folks will love - something that feels like talking to a friend who really knows their stuff.

What we need: {prompt}

Content:"""
        ],
        "Witty": [
            """Create clever, engaging {channel} content for {industry} that:
- Uses humor and wit appropriately
- Includes playful language and creative metaphors
- Stands out from boring industry content
- Makes people smile while informing them

Request: {prompt}

Witty {channel} content:""",
            """Time to shake things up! Create some witty {channel} content that makes {industry} actually fun and memorable.

Challenge: {prompt}

Creative content:"""
        ]
    }
    
    def __init__(self, config: Optional[GenerationConfig] = None, model_path: Optional[str] = None):
        self.config = config or GenerationConfig()
        self.evaluator = ModelEvaluator()
        
        # Get Hugging Face token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set. Please set it with your Hugging Face token.")
        
        # Initialize model and tokenizer
        self.model_name = model_path if model_path else self.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {self.model_name} on {self.device}...")
        self._load_model_and_tokenizer()
        print("Model loaded successfully!")
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with optimized settings"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=self.hf_token,
            padding_side="left"  # Better for generation
        )
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            token=self.hf_token,
            device_map="auto",
            load_in_8bit=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(self.device)
    
    def generate_content(
        self, 
        prompt: str,
        tone: str = "Professional",
        industry: Optional[str] = None,
        channel: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate content based on prompt and tone with enhanced prompting."""
        # Validate tone
        if tone not in self.TONES:
            print(f"Warning: Invalid tone '{tone}'. Using 'Professional' instead.")
            tone = "Professional"
        
        # Use provided config or default
        gen_config = config or self.config
        
        # Construct enhanced prompt
        formatted_prompt = self._construct_enhanced_prompt(prompt, tone, industry, channel)
        
        # Set generation parameters
        generation_params = {
            "max_length": gen_config.max_length,
            "max_new_tokens": min(512, gen_config.max_length // 2),  # Limit new tokens
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "do_sample": gen_config.do_sample,
            "repetition_penalty": gen_config.repetition_penalty,
            "length_penalty": gen_config.length_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Encode prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = input_ids.to(self.device)
        
        # Generate content
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                **generation_params
            )
        
        # Decode and clean output
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text and clean
        result = generated_text[len(formatted_prompt):].strip()
        result = self._post_process_output(result, tone)
        
        return result
    
    def _construct_enhanced_prompt(
        self, 
        prompt: str, 
        tone: str = "Professional",
        industry: Optional[str] = None,
        channel: Optional[str] = None
    ) -> str:
        """Construct enhanced prompt with better context and instructions."""
        # Set defaults
        industry = industry or "general business"
        channel = channel or "marketing content"
        
        # Select appropriate template
        templates = self.PROMPT_TEMPLATES.get(tone, self.PROMPT_TEMPLATES["Professional"])
        template = random.choice(templates)
        
        # Format template
        formatted_prompt = template.format(
            prompt=prompt,
            industry=industry,
            channel=channel
        )
        
        return formatted_prompt
    
    def _post_process_output(self, text: str, tone: str) -> str:
        """Post-process generated text to improve quality"""
        # Remove common artifacts
        text = text.replace("<|endoftext|>", "").strip()
        
        # Remove incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            text = '.'.join(sentences[:-1]) + '.'
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Add appropriate punctuation if missing
        if text and not text.endswith(('.', '!', '?')):
            if tone == "Casual" or tone == "Witty":
                text += "!"
            else:
                text += "."
        
        return text
    
    def fine_tune_model(
        self, 
        dataset_path: str, 
        output_dir: Optional[str] = None,
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        merge_weights: bool = True,
        evaluation_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced fine-tuning with evaluation and monitoring."""
        # Set output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(settings.MODEL_DIR, "mistral-7b-marketing-lora-v2" if use_lora else "mistral-7b-marketing-v2")
        
        # Ensure model directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare dataset with validation split
        train_dataset, eval_dataset = self._prepare_dataset_with_split(dataset_path, evaluation_data)
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        if use_lora:
            # Enhanced LoRA config
            if lora_config is None:
                lora_config = {
                    "r": 32,  # Increased rank for better capacity
                    "lora_alpha": 64,  # Increased alpha
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "task_type": TaskType.CAUSAL_LM,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More modules
                    "modules_to_save": None,
                }
            
            # Create LoRA config
            peft_config = LoraConfig(**lora_config)
            
            # Get PEFT model
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        # Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,  # Increased epochs
            per_device_train_batch_size=2,  # Larger batch size
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,  # Better effective batch size
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=250 if eval_dataset else None,
            save_steps=500,
            save_total_limit=3,
            logging_strategy="steps",
            logging_steps=50,
            learning_rate=1e-4,  # Lower learning rate for stability
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            fp16=self.device == "cuda",
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to=["tensorboard"],
            run_name=f"mistral-marketing-{locals().get('tone', 'multi')}-{self.MODEL_VERSION}",
            gradient_checkpointing=True,  # Save memory
            dataloader_num_workers=2,
            seed=42,  # Reproducibility
        )
        
        # Enhanced data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # Optimize for tensor cores
        )
        
        # Setup trainer with callbacks
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )
        
        # Train the model
        print("Starting training...")
        training_result = trainer.train()
        
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
        
        # Save training configuration
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "model_version": self.MODEL_VERSION,
                "lora_config": lora_config if use_lora else None,
                "training_args": training_args.to_dict(),
                "dataset_path": dataset_path,
                "use_lora": use_lora,
                "merge_weights": merge_weights,
            }, f, indent=2)
        
        print(f"Training completed! Model saved to {output_dir}")
        return {
            "training_loss": training_result.training_loss,
            "training_steps": training_result.global_step,
            "output_dir": output_dir,
            "config_saved": config_path
        }
    
    def _prepare_dataset_with_split(self, dataset_path: str, evaluation_data: Optional[str] = None):
        """Prepare dataset with train/eval split"""
        # Load CSV dataset
        df = pd.read_csv(dataset_path)
        
        # Check required columns
        if "prompt" not in df.columns or "content" not in df.columns:
            raise ValueError("Dataset missing required columns: prompt, content")
        
        # Prepare training data
        train_texts = self._format_training_texts(df)
        
        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts})
        
        # Handle evaluation data
        eval_dataset = None
        if evaluation_data and os.path.exists(evaluation_data):
            eval_df = pd.read_csv(evaluation_data)
            eval_texts = self._format_training_texts(eval_df)
            eval_dataset = Dataset.from_dict({"text": eval_texts})
        elif len(train_texts) > 100:  # Split if enough data
            # Split dataset
            split_idx = int(0.9 * len(train_texts))
            eval_texts = train_texts[split_idx:]
            train_texts = train_texts[:split_idx]
            
            train_dataset = Dataset.from_dict({"text": train_texts})
            eval_dataset = Dataset.from_dict({"text": eval_texts})
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=1024,
                return_tensors="pt"
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        if eval_dataset:
            eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return train_dataset, eval_dataset
    
    def _format_training_texts(self, df: pd.DataFrame) -> List[str]:
        """Format training texts with enhanced prompts"""
        texts = []
        for _, row in df.iterrows():
            prompt = row["prompt"]
            content = row["content"]
            
            # Get metadata if available
            metadata = {}
            if "metadata" in df.columns and pd.notna(row["metadata"]):
                try:
                    metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
                except json.JSONDecodeError:
                    pass
            
            # Create enhanced prompt
            formatted_prompt = self._construct_enhanced_prompt(
                prompt, 
                metadata.get("tone", "Professional"),
                metadata.get("industry"),
                metadata.get("channel")
            )
            
            # Combine prompt and content with special tokens
            full_text = f"{formatted_prompt}\n\n{content}{self.tokenizer.eos_token}"
            texts.append(full_text)
        
        return texts
    
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
    
    def evaluate_model(self, test_prompts: List[Dict], reference_outputs: List[str]) -> Dict:
        """Evaluate model performance"""
        return self.evaluator.evaluate_generation_quality(self, test_prompts, reference_outputs)
    
    def save_generated_content(
        self,
        db: Session,
        content: str,
        prompt: str,
        tone: str,
        user_id: int,
        channel: Optional[str] = None,
        industry: Optional[str] = None,
        evaluation_scores: Optional[Dict] = None,
    ) -> Content:
        """Save generated content to database with evaluation scores."""
        generation_params = {
            "prompt": prompt,
            "tone": tone,
            "industry": industry,
            "channel": channel,
            "model_version": self.MODEL_VERSION,
            "generation_config": self.config.__dict__,
        }
        
        if evaluation_scores:
            generation_params["evaluation_scores"] = evaluation_scores
        
        # Create content object
        db_content = Content(
            title=prompt[:100] if prompt else f"Generated content with {tone} tone",
            body=content,
            type=channel.lower() if channel else "general",
            status="draft",
            ai_generated=True,
            generation_params=generation_params,
            model_version=self.MODEL_VERSION,
            created_by_id=user_id,
            target_audience=None,
        )
        
        db.add(db_content)
        db.commit()
        db.refresh(db_content)
        
        return db_content