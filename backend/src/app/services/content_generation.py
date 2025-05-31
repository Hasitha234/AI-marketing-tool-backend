# app/services/gemini_content_generation.py

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

import google.genai as genai
from google.genai.types import HarmCategory, HarmBlockThreshold, SafetySetting, GenerateContentConfig
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.crud import content as content_crud
from app.schemas.content import ContentCreate
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeminiGenerationConfig:
    """Configuration for Gemini 2.0 content generation parameters"""
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 1024
    candidate_count: int = 1
    
    # Safety settings for Gemini 2.0
    safety_settings: List[SafetySetting] = None
    
    def __post_init__(self):
        if self.safety_settings is None:
            self.safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
            ]

class GeminiContentGenerationService:
    """Content generation service using Google's Gemini 2.0 Flash API"""
    
    def __init__(self, api_key: str, config: GeminiGenerationConfig = None):
        self.api_key = api_key
        self.config = config or GeminiGenerationConfig()
        self.client = None
        
        # Define available options for the UI (matching your existing structure)
        self.TONES = [
            "professional", "casual", "friendly", "authoritative",
            "conversational", "enthusiastic", "informative", "persuasive",
            "formal", "creative", "technical", "empathetic"
        ]
        
        self.CHANNELS = [
            "blog", "ad_copy", "social", "email", "website", "newsletter",
            "press_release", "product_description", "landing_page", "article"
        ]
        
        self.INDUSTRIES = [
            "technology", "health_fitness", "business_finance", 
            "lifestyle_travel", "education_career", "retail",
            "healthcare", "finance", "real_estate", "automotive",
            "food_beverage", "entertainment", "non_profit", "consulting"
        ]
        
        # Content type specific configurations
        self.content_configs = {
            'blog': {
                'max_tokens': 800,
                'structure': 'introduction, main points, conclusion',
                'style': 'informative and engaging'
            },
            'ad_copy': {
                'max_tokens': 150,
                'structure': 'hook, benefit, call-to-action',
                'style': 'persuasive and compelling'
            },
            'social': {
                'max_tokens': 100,
                'structure': 'engaging hook, key message, hashtags',
                'style': 'conversational and shareable'
            },
            'email': {
                'max_tokens': 300,
                'structure': 'subject line, greeting, body, call-to-action',
                'style': 'personal and direct'
            },
            'website': {
                'max_tokens': 400,
                'structure': 'clear sections, benefits, action items',
                'style': 'professional and clear'
            },
            'newsletter': {
                'max_tokens': 500,
                'structure': 'headline, key updates, resources, conclusion',
                'style': 'informative and engaging'
            }
        }
        
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini 2.0 client"""
        try:
            # Initialize the client with API key
            self.client = genai.Client(api_key=self.api_key)
            
            logger.info("Gemini 2.0 Flash API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini 2.0 API: {str(e)}")
            raise RuntimeError(f"Gemini initialization error: {str(e)}")
    
    def _build_prompt(self, keywords: str, content_type: str, tone: str, 
                     industry: Optional[str] = None, additional_context: Optional[str] = None) -> str:
        """Build optimized prompt for content generation"""
        
        content_config = self.content_configs.get(content_type, self.content_configs['blog'])
        
        # Base prompt structure
        prompt_parts = [
            f"You are an expert content writer specializing in {industry or 'general'} industry.",
            f"Create {content_type} content with a {tone} tone.",
            f"Target keywords: {keywords}",
            "",
            f"Content requirements:",
            f"- Type: {content_type}",
            f"- Tone: {tone}",
            f"- Style: {content_config['style']}",
            f"- Structure: {content_config['structure']}",
            f"- Maximum length: approximately {content_config['max_tokens']} tokens",
        ]
        
        if industry:
            prompt_parts.append(f"- Industry focus: {industry}")
        
        if additional_context:
            prompt_parts.append(f"- Additional context: {additional_context}")
        
        prompt_parts.extend([
            "",
            "Guidelines:",
            "- Write engaging, original content",
            "- Include the target keywords naturally",
            "- Ensure content is valuable to the target audience",
            "- Use appropriate formatting and structure",
            "- Make it actionable and relevant",
            "",
            "Generate the content now:"
        ])
        
        return "\n".join(prompt_parts)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate_content_async(self, keywords: str, content_type: str = "blog", 
                                   tone: str = "professional", industry: Optional[str] = None,
                                   additional_context: Optional[str] = None) -> str:
        """Generate content asynchronously with retry logic using Gemini 2.0"""
        try:
            if not keywords or not keywords.strip():
                raise ValueError("Keywords cannot be empty")
            
            # Validate inputs
            tone = tone if tone in self.TONES else "professional"
            content_type = content_type if content_type in self.CHANNELS else "blog"
            
            # Build the prompt
            prompt = self._build_prompt(keywords, content_type, tone, industry, additional_context)
            
            # Update generation config for specific content type
            content_config = self.content_configs.get(content_type, self.content_configs['blog'])
            
            # Create generation config for Gemini 2.0
            generation_config = GenerateContentConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=min(content_config['max_tokens'], self.config.max_output_tokens),
                candidate_count=1,
                safety_settings=self.config.safety_settings
            )
            
            # Generate content using Gemini 2.0 Flash
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash-exp",  # Updated to Gemini 2.0 Flash
                contents=prompt,
                config=generation_config
            )
            
            # Extract the generated text
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    generated_text = candidate.content.parts[0].text
                    
                    # Clean and validate the generated content
                    cleaned_content = self._clean_generated_content(generated_text, content_type)
                    
                    logger.info(f"Successfully generated {len(cleaned_content.split())} words for {content_type}")
                    return cleaned_content
                else:
                    raise RuntimeError("No text content in response")
            else:
                raise RuntimeError("No candidates in response from Gemini 2.0 API")
                
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise RuntimeError(f"Content generation error: {str(e)}")
    
    def generate_content(self, keywords: str, content_type: str = "blog", 
                        tone: str = "professional", industry: Optional[str] = None,
                        additional_context: Optional[str] = None) -> str:
        """Synchronous wrapper for content generation"""
        return asyncio.run(self.generate_content_async(
            keywords, content_type, tone, industry, additional_context
        ))
    
    def _clean_generated_content(self, content: str, content_type: str) -> str:
        """Clean and format generated content"""
        if not content:
            return content
        
        # Remove common artifacts from AI generation
        content = content.strip()
        
        # Remove any instruction-like text at the beginning
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like instructions or metadata
            if not (line.startswith(('Here is', 'Here\'s', 'Based on', 'According to')) and len(clean_lines) == 0):
                clean_lines.append(line)
        
        content = '\n'.join(clean_lines).strip()
        
        # Ensure content ends properly for certain types
        if content_type in ['blog', 'article', 'newsletter']:
            content = self._ensure_complete_sentences(content)
        
        return content
    
    def _ensure_complete_sentences(self, content: str) -> str:
        """Ensure content ends with complete sentences"""
        if not content:
            return content
        
        content = content.strip()
        sentence_endings = ['.', '!', '?']
        
        # If content doesn't end with proper punctuation, try to fix it
        if content and content[-1] not in sentence_endings:
            # Find the last complete sentence
            sentences = []
            for ending in sentence_endings:
                sentences.extend(content.split(ending))
            
            # Reconstruct with complete sentences only
            if len(sentences) > 1:
                complete_parts = []
                for i, sentence in enumerate(sentences[:-1]):  # Exclude the last incomplete part
                    if sentence.strip():
                        complete_parts.append(sentence.strip())
                
                if complete_parts:
                    # Find the original ending punctuation
                    for ending in sentence_endings:
                        if ending in content:
                            return '. '.join(complete_parts) + '.'
        
        return content
    
    async def batch_generate_content_async(self, prompt_configs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Generate content for multiple prompts asynchronously"""
        tasks = []
        
        for config in prompt_configs:
            task = self.generate_content_async(
                keywords=config.get("keywords", config.get("prompt", "")),
                content_type=config.get("content_type", config.get("channel", "blog")),
                tone=config.get("tone", "professional"),
                industry=config.get("industry"),
                additional_context=config.get("additional_context")
            )
            tasks.append((task, config))
        
        results = []
        for task, config in tasks:
            try:
                generated_content = await task
                results.append({
                    "status": "success",
                    "keywords": config.get("keywords", config.get("prompt")),
                    "content_type": config.get("content_type", config.get("channel")),
                    "tone": config.get("tone"),
                    "industry": config.get("industry"),
                    "generated_content": generated_content,
                    "word_count": len(generated_content.split()),
                    "character_count": len(generated_content),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "keywords": config.get("keywords", config.get("prompt")),
                    "content_type": config.get("content_type", config.get("channel")),
                    "tone": config.get("tone"),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def batch_generate_content(self, prompt_configs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for batch content generation"""
        return asyncio.run(self.batch_generate_content_async(prompt_configs))
    
    def save_generated_content(
        self,
        db: Session,
        content: str,
        prompt: str,
        tone: str,
        user_id: int,
        channel: Optional[str] = None,
        industry: Optional[str] = None,
        additional_metadata: Optional[Dict] = None
    ):
        """Save generated content to database"""
        try:
            metadata = {
                "prompt": prompt,
                "tone": tone,
                "industry": industry,
                "channel": channel,
                "word_count": len(content.split()),
                "character_count": len(content),
                "generated_at": datetime.now().isoformat(),
                "api_provider": "gemini",
                "model_version": "gemini-2.0-flash-exp"
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            content_data = ContentCreate(
                title=f"Generated {channel or 'content'} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                body=content,
                type=channel or "general",
                status="draft",
                ai_generated=True,
                created_by_id=user_id,
                metadata=metadata
            )
            
            saved_content = content_crud.create_content(db=db, content_in=content_data, user_id=user_id)
            logger.info(f"Content saved with ID: {saved_content.id}")
            return saved_content
            
        except Exception as e:
            logger.error(f"Failed to save content: {str(e)}")
            raise RuntimeError(f"Failed to save content: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "Google Gemini 2.0 Flash",
            "api_provider": "Google AI",
            "model_version": "gemini-2.0-flash-exp",
            "available_tones": self.TONES,
            "available_channels": self.CHANNELS,
            "available_industries": self.INDUSTRIES,
            "content_configs": self.content_configs,
            "generation_config": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_output_tokens
            }
        }
    
    async def health_check_async(self) -> bool:
        """Check if the service is healthy and ready to generate content"""
        try:
            if self.client is None:
                logger.error("Health check failed: Client is None")
                return False
            
            # Simple test generation with minimal content
            test_response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents="Say 'OK'",
                config=GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=5
                )
            )
            
            # Check if we got a valid response
            if (test_response and 
                hasattr(test_response, 'candidates') and 
                test_response.candidates and 
                len(test_response.candidates) > 0):
                
                candidate = test_response.candidates[0]
                if (hasattr(candidate, 'content') and 
                    candidate.content and 
                    hasattr(candidate.content, 'parts') and
                    candidate.content.parts and
                    len(candidate.content.parts) > 0):
                    
                    text_content = candidate.content.parts[0].text
                    if text_content and len(text_content.strip()) > 0:
                        logger.info("Health check passed: Gemini 2.0 API responding correctly")
                        return True
            
            logger.error("Health check failed: Invalid response structure")
            return False
            
        except Exception as e:
            logger.error(f"Health check failed with exception: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            return False
    
    def health_check(self) -> bool:
        """Synchronous wrapper for health check"""
        try:
            # Check if we're in an existing event loop
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, create a task
                if loop.is_running():
                    # We're in an async context, so we need to handle this differently
                    return self._sync_health_check()
            except RuntimeError:
                # No running loop, we can create a new one
                pass
            
            return asyncio.run(self.health_check_async())
        except Exception as e:
            logger.error(f"Health check wrapper failed: {str(e)}")
            return False
    
    def _sync_health_check(self) -> bool:
        """Synchronous health check that doesn't use async/await"""
        try:
            if self.client is None:
                return False
            
            # Simple sync check - just verify client initialization
            # We can't do actual API calls in sync context without proper async handling
            return True
            
        except Exception as e:
            logger.error(f"Sync health check failed: {str(e)}")
            return False

# Global service instance
_gemini_service: Optional[GeminiContentGenerationService] = None

def get_gemini_content_generation_service(
    api_key: Optional[str] = None,
    config: Optional[GeminiGenerationConfig] = None
) -> GeminiContentGenerationService:
    """Get or create Gemini content generation service instance"""
    global _gemini_service
    
    if _gemini_service is None:
        if api_key is None:
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("Gemini API key is required")
        
        if config is None:
            config = GeminiGenerationConfig()
        
        _gemini_service = GeminiContentGenerationService(api_key, config)
        
        # Perform health check
        if not _gemini_service.health_check():
            logger.warning("Gemini content generation service health check failed")
    
    return _gemini_service

def reset_gemini_content_generation_service():
    """Reset the global service instance (useful for testing)"""
    global _gemini_service
    _gemini_service = None