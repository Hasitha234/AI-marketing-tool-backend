# scripts/setup_dialogflow.py
"""
Script to set up Dialogflow intents and entities for the AI Marketing Tool chatbot
Run this after creating your Dialogflow project and setting up authentication
"""

import os
import json
from google.cloud import dialogflow
from typing import List, Dict, Any
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Debug information
print("Debug: Current working directory:", os.getcwd())
print("Debug: GOOGLE_CLOUD_PROJECT_ID:", os.getenv('GOOGLE_CLOUD_PROJECT_ID'))
print("Debug: GOOGLE_APPLICATION_CREDENTIALS:", os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

class DialogflowSetup:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.intents_client = dialogflow.IntentsClient()
        self.entity_types_client = dialogflow.EntityTypesClient()
        self.parent = f"projects/{project_id}/agent"
    
    def create_intent(self, display_name: str, training_phrases: List[str], 
                     response_messages: List[str], parameters: List[Dict] = None):
        """Create a new intent in Dialogflow"""
        
        # Create training phrases
        training_phrases_list = []
        for phrase in training_phrases:
            parts = [dialogflow.Intent.TrainingPhrase.Part(text=phrase)]
            training_phrases_list.append(
                dialogflow.Intent.TrainingPhrase(parts=parts)
            )
        
        # Create response messages
        messages = []
        for message in response_messages:
            messages.append(
                dialogflow.Intent.Message(
                    text=dialogflow.Intent.Message.Text(text=[message])
                )
            )
        
        # Create intent
        intent = dialogflow.Intent(
            display_name=display_name,
            training_phrases=training_phrases_list,
            messages=messages
        )
        
        # Add parameters if provided
        if parameters:
            intent.parameters = []
            for param in parameters:
                intent.parameters.append(
                    dialogflow.Intent.Parameter(
                        display_name=param['display_name'],
                        entity_type_display_name=param['entity_type'],
                        mandatory=param.get('mandatory', False),
                        prompts=param.get('prompts', [])
                    )
                )
        
        # Create the intent
        response = self.intents_client.create_intent(
            parent=self.parent, intent=intent
        )
        
        print(f"Intent created: {response.display_name}")
        return response
    
    def create_entity_type(self, display_name: str, entities: List[Dict[str, List[str]]]):
        """Create a new entity type in Dialogflow"""
        
        entity_list = []
        for entity_data in entities:
            for value, synonyms in entity_data.items():
                entity_list.append(
                    dialogflow.EntityType.Entity(
                        value=value,
                        synonyms=synonyms
                    )
                )
        
        entity_type = dialogflow.EntityType(
            display_name=display_name,
            kind=dialogflow.EntityType.Kind.KIND_MAP,
            entities=entity_list
        )
        
        response = self.entity_types_client.create_entity_type(
            parent=self.parent, entity_type=entity_type
        )
        
        print(f"Entity type created: {response.display_name}")
        return response
    
    def setup_all_intents(self):
        """Set up all intents for the AI Marketing Tool chatbot"""
        
        # 1. Welcome Intent (usually exists by default)
        print("Setting up Welcome Intent...")
        
        # 2. FAQ - Pricing Intent
        print("Creating FAQ - Pricing Intent...")
        self.create_intent(
            display_name="FAQ - Pricing",
            training_phrases=[
                "What are your pricing plans?",
                "How much does it cost?",
                "Tell me about pricing",
                "What's the price?",
                "How much do I need to pay?",
                "Cost information",
                "Pricing details",
                "What are the rates?",
                "How expensive is it?",
                "Price list"
            ],
            response_messages=[
                "Let me tell you about our flexible pricing plans for the AI Marketing Platform!"
            ]
        )
        
        # 3. FAQ - Features Intent
        print("Creating FAQ - Features Intent...")
        self.create_intent(
            display_name="FAQ - Features",
            training_phrases=[
                "What features do you offer?",
                "Tell me about your features",
                "What can your platform do?",
                "What functionality is available?",
                "Feature list",
                "What capabilities do you have?",
                "Show me what you can do",
                "Platform features",
                "What's included?",
                "Tool capabilities"
            ],
            response_messages=[
                "Our AI Marketing Platform includes powerful features for lead scoring, content generation, and automation!"
            ]
        )
        
        # 4. FAQ - Support Intent
        print("Creating FAQ - Support Intent...")
        self.create_intent(
            display_name="FAQ - Support",
            training_phrases=[
                "How can I get support?",
                "I need help",
                "Customer support",
                "Technical support",
                "Help desk",
                "Contact support",
                "Support options",
                "Get assistance",
                "How to get help?",
                "Support team"
            ],
            response_messages=[
                "We offer comprehensive support through multiple channels to help you succeed!"
            ]
        )
        
        # 5. Schedule Appointment Intent
        print("Creating Schedule Appointment Intent...")
        self.create_intent(
            display_name="Schedule Appointment",
            training_phrases=[
                "I want to schedule a demo",
                "Book a consultation",
                "Schedule a meeting",
                "I'd like to set up an appointment",
                "Can we schedule a call?",
                "Book a demo",
                "Arrange a consultation",
                "Set up a meeting",
                "Schedule time to talk",
                "I want to book a session"
            ],
            response_messages=[
                "I'd be happy to help you schedule a consultation!"
            ],
            parameters=[
                {
                    'display_name': 'consultation-type',
                    'entity_type': '@consultation-type',
                    'mandatory': False
                },
                {
                    'display_name': 'date',
                    'entity_type': '@sys.date',
                    'mandatory': False
                },
                {
                    'display_name': 'time',
                    'entity_type': '@sys.time',
                    'mandatory': False
                }
            ]
        )
        
        # 6. Product Recommendation Intent
        print("Creating Product Recommendation Intent...")
        self.create_intent(
            display_name="Product Recommendation",
            training_phrases=[
                "Which plan is right for me?",
                "What do you recommend?",
                "Help me choose a plan",
                "What's the best option for my business?",
                "Recommend a solution",
                "Which package should I get?",
                "What fits my needs?",
                "Best plan for small business",
                "Enterprise solution recommendation",
                "Help me decide"
            ],
            response_messages=[
                "I'd be happy to recommend the best plan for your business needs!"
            ],
            parameters=[
                {
                    'display_name': 'company-size',
                    'entity_type': '@company-size',
                    'mandatory': False
                },
                {
                    'display_name': 'budget',
                    'entity_type': '@budget-range',
                    'mandatory': False
                },
                {
                    'display_name': 'industry',
                    'entity_type': '@industry',
                    'mandatory': False
                }
            ]
        )
        
        # 7. Lead Qualification Intent
        print("Creating Lead Qualification Intent...")
        self.create_intent(
            display_name="Lead Qualification",
            training_phrases=[
                "Tell me about your company",
                "I'm interested in your platform",
                "We're looking for a marketing solution",
                "Our company needs help with leads",
                "We want to improve our marketing",
                "Looking for marketing automation",
                "Need help with lead generation",
                "Interested in AI marketing tools",
                "Want to qualify for your service",
                "We're a potential customer"
            ],
            response_messages=[
                "Great! I'd love to learn more about your business to see how we can help."
            ],
            parameters=[
                {
                    'display_name': 'company-size',
                    'entity_type': '@company-size',
                    'mandatory': False
                },
                {
                    'display_name': 'industry',
                    'entity_type': '@industry',
                    'mandatory': False
                },
                {
                    'display_name': 'budget',
                    'entity_type': '@budget-range',
                    'mandatory': False
                }
            ]
        )
        
        # 8. Contact Sales Intent
        print("Creating Contact Sales Intent...")
        self.create_intent(
            display_name="Contact Sales",
            training_phrases=[
                "I want to talk to sales",
                "Connect me with sales team",
                "Speak to a sales representative",
                "Contact sales",
                "Sales team",
                "Talk to someone from sales",
                "I need to speak with sales",
                "Put me in touch with sales",
                "Sales contact",
                "Reach out to sales"
            ],
            response_messages=[
                "I'll connect you with our sales team right away!"
            ]
        )
        
        print("All intents created successfully!")
    
    def setup_entity_types(self):
        """Set up entity types for parameters"""
        
        print("Creating entity types...")
        
        # Company Size Entity
        self.create_entity_type(
            display_name="company-size",
            entities=[
                {"startup": ["startup", "start-up", "new company", "small startup"]},
                {"small": ["small", "small business", "small company", "SMB", "1-10 employees"]},
                {"medium": ["medium", "medium-sized", "mid-size", "10-100 employees", "growing company"]},
                {"large": ["large", "big company", "enterprise", "100+ employees", "corporation"]},
                {"enterprise": ["enterprise", "large enterprise", "Fortune 500", "multinational", "global company"]}
            ]
        )
        
        # Budget Range Entity
        self.create_entity_type(
            display_name="budget-range",
            entities=[
                {"low": ["under $100", "less than $100", "budget-friendly", "cheap", "affordable"]},
                {"medium": ["$100-$500", "$100 to $500", "moderate budget", "reasonable price"]},
                {"high": ["$500+", "over $500", "premium", "enterprise budget", "flexible budget"]},
                {"custom": ["custom pricing", "enterprise pricing", "contact for pricing", "negotiable"]}
            ]
        )
        
        # Industry Entity
        self.create_entity_type(
            display_name="industry",
            entities=[
                {"technology": ["technology", "tech", "software", "IT", "SaaS"]},
                {"healthcare": ["healthcare", "medical", "health", "pharma", "biotech"]},
                {"finance": ["finance", "financial services", "banking", "fintech", "insurance"]},
                {"retail": ["retail", "e-commerce", "online store", "shopping", "consumer goods"]},
                {"manufacturing": ["manufacturing", "industrial", "factory", "production"]},
                {"education": ["education", "school", "university", "training", "learning"]},
                {"marketing": ["marketing", "advertising", "digital marketing", "agency"]},
                {"other": ["other", "different industry", "various", "mixed"]}
            ]
        )
        
        # Consultation Type Entity
        self.create_entity_type(
            display_name="consultation-type",
            entities=[
                {"demo": ["demo", "demonstration", "product demo", "show me the platform"]},
                {"technical": ["technical", "technical discussion", "technical consultation", "implementation"]},
                {"pricing": ["pricing", "pricing discussion", "cost discussion", "budget planning"]},
                {"general": ["general", "general consultation", "overview", "introduction"]},
                {"training": ["training", "onboarding", "setup help", "learning session"]}
            ]
        )
        
        print("Entity types created successfully!")


def main():
    """Main function to set up Dialogflow intents and entities"""
    
    # Get project ID from environment variable
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    if not project_id:
        print("Please set GOOGLE_CLOUD_PROJECT_ID environment variable")
        return
    
    # Initialize setup
    setup = DialogflowSetup(project_id)
    
    try:
        # Create entity types first
        setup.setup_entity_types()
        
        # Then create intents
        setup.setup_all_intents()
        
        print("\n✅ Dialogflow setup completed successfully!")
        print("Your chatbot is now ready to handle:")
        print("- Welcome messages")
        print("- FAQ about pricing, features, and support")
        print("- Appointment scheduling")
        print("- Product recommendations")
        print("- Lead qualification")
        print("- Sales team contact requests")
        
    except Exception as e:
        print(f"❌ Error setting up Dialogflow: {str(e)}")
        print("Make sure you have:")
        print("1. Created a Dialogflow project")
        print("2. Enabled the Dialogflow API")
        print("3. Set up authentication credentials")
        print("4. Set the GOOGLE_CLOUD_PROJECT_ID environment variable")


if __name__ == "__main__":
    main()


