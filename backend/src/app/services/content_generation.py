import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
from sqlalchemy.orm import Session
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for content generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_length: int = 1024
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512

class ContentGenerationService:
    """Service for generating marketing content using fine-tuned Zephyr model."""
    
    MODEL_VERSION = "zephyr-7b-marketing-v1.0"
    
    # Available tones for the UI
    TONES = [
        "Professional", "Casual", "Energetic", "Friendly", "Eco-conscious",
        "Authoritative", "Conversational", "Persuasive", "Informative", "Creative"
    ]
    
    # Available channels
    CHANNELS = [
        "Blog", "Social Media", "Email", "Website", "Advertisement", 
        "Newsletter", "Press Release", "Product Description"
    ]
    
    # Available industries
    INDUSTRIES = [
        "Technology", "Fashion", "Fitness", "Food & Beverage", "Healthcare",
        "Finance", "Education", "Real Estate", "Automotive", "Travel"
    ]

    def __init__(self, model_path: Optional[str] = None, config: Optional[GenerationConfig] = None):
        """Initialize the content generation service."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.config = config or GenerationConfig()
        
        # Set model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = Path(__file__).parent.parent.parent / "content_model" / "zephyr-marketing-lora"
            
        self._load_model()

    def _load_model(self):
        """Load the fine-tuned Zephyr model with LoRA adapters."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Base model configuration
            base_model_name = "HuggingFaceH4/zephyr-7b-beta"
            
            # Load tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load base model with optimizations
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Check if LoRA adapters exist
            if self.model_path.exists() and (self.model_path / "adapter_model.safetensors").exists():
                logger.info("Loading LoRA adapters...")
                
                # Load the fine-tuned model with LoRA adapters
                self.model = PeftModel.from_pretrained(
                    base_model,
                    str(self.model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                logger.info("LoRA adapters loaded successfully")
                
            else:
                logger.warning("LoRA adapters not found, using base model")
                self.model = base_model
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _format_prompt_for_zephyr(self, prompt: str, tone: str, industry: Optional[str] = None, 
                                  channel: Optional[str] = None) -> str:
        """Format the prompt according to Zephyr's chat template."""
        
        # Build instruction
        instruction_parts = [f"Content brief: {prompt}"]
        
        if tone:
            instruction_parts.append(f"Tone: {tone}")
        if channel:
            instruction_parts.append(f"Channel: {channel}")
        if industry:
            instruction_parts.append(f"Industry: {industry}")
        
        instruction = "Generate marketing content with the following specifications: " + "; ".join(instruction_parts)
        
        # System message
        system_message = "You are a helpful AI assistant specialized in creating high-quality marketing content. Generate engaging, persuasive, and brand-appropriate content based on the given specifications."
        
        # Format in Zephyr's chat format
        formatted_prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n"
        
        return formatted_prompt

    def generate_content(self, prompt: str, tone: str = "Professional", 
                        industry: Optional[str] = None, channel: Optional[str] = None,
                        max_length: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Generate marketing content using the fine-tuned model."""
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Please check model initialization.")
        
        try:
            # Use provided parameters or defaults
            max_length = max_length or self.config.max_length
            temperature = temperature or self.config.temperature
            
            # Format prompt for Zephyr
            formatted_prompt = self._format_prompt_for_zephyr(prompt, tone, industry, channel)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length - self.config.max_new_tokens,  # Leave room for generation
                padding=False
            )
            
            # Move inputs to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Generate content
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|assistant|>" in generated_text:
                assistant_response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                # Fallback: remove the input prompt from the output
                assistant_response = generated_text[len(formatted_prompt):].strip()
            
            # Clean up the response (remove any remaining special tokens or artifacts)
            assistant_response = self._clean_generated_content(assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise RuntimeError(f"Content generation failed: {str(e)}")

    def _clean_generated_content(self, content: str) -> str:
        """Clean up generated content by removing artifacts and improving formatting."""
        
        # Remove common artifacts
        artifacts_to_remove = [
            "</s>", "<s>", "<|user|>", "<|assistant|>", "<|system|>",
            "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"
        ]
        
        for artifact in artifacts_to_remove:
            content = content.replace(artifact, "")
        
        # Clean up extra whitespace and newlines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        content = '\n'.join(lines)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        return content

    def save_generated_content(self, db: Session, content: str, prompt: str, 
                             tone: str, user_id: int, channel: Optional[str] = None, 
                             industry: Optional[str] = None) -> Any:
        """Save generated content to the database."""
        
        from app.models.content import Content
        
        # Create content object
        db_content = Content(
            title=prompt[:100] if len(prompt) > 100 else prompt,
            body=content,
            type=channel or "General",
            ai_generated=True,
            model_version=self.MODEL_VERSION,
            generation_params={
                "prompt": prompt,
                "tone": tone,
                "industry": industry,
                "channel": channel,
                "model_version": self.MODEL_VERSION,
                "generation_config": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_length": self.config.max_length
                }
            },
            created_by_id=user_id
        )
        
        # Save to database
        db.add(db_content)
        db.commit()
        db.refresh(db_content)
        
        return db_content

    def batch_generate_content(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate content for multiple prompts in batch."""
        
        results = []
        
        for i, prompt_config in enumerate(prompts):
            try:
                logger.info(f"Generating content for prompt {i+1}/{len(prompts)}")
                
                content = self.generate_content(
                    prompt=prompt_config.get('prompt', ''),
                    tone=prompt_config.get('tone', 'Professional'),
                    industry=prompt_config.get('industry'),
                    channel=prompt_config.get('channel')
                )
                
                result = {
                    **prompt_config,
                    'generated_content': content,
                    'status': 'success',
                    'content_length': len(content),
                    'word_count': len(content.split())
                }
                
            except Exception as e:
                logger.error(f"Failed to generate content for prompt {i+1}: {str(e)}")
                result = {
                    **prompt_config,
                    'generated_content': '',
                    'status': 'failed',
                    'error': str(e),
                    'content_length': 0,
                    'word_count': 0
                }
            
            results.append(result)
        
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        
        if not self.model:
            return {"status": "not_loaded"}
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "loaded",
            "model_version": self.MODEL_VERSION,
            "base_model": "HuggingFaceH4/zephyr-7b-beta",
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_path": str(self.model_path),
            "available_tones": self.TONES,
            "available_channels": self.CHANNELS,
            "available_industries": self.INDUSTRIES
        }

# Create singleton instance (will be initialized when imported)
_content_generation_service = None

def get_content_generation_service(model_path: Optional[str] = None, 
                                  config: Optional[GenerationConfig] = None) -> ContentGenerationService:
    """Get the singleton content generation service."""
    global _content_generation_service
    
    if _content_generation_service is None:
        _content_generation_service = ContentGenerationService(model_path, config)
    
    return _content_generation_service