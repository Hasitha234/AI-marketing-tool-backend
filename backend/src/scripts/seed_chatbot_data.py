"""
Script to seed the database with initial chatbot data (FAQs, etc.)
"""

from app.db.session import SessionLocal
from app.crud.chatbot import faq_crud
from app.schemas.chatbot import FAQCreate


def seed_faqs():
    """Seed FAQ data"""
    db = SessionLocal()
    
    faqs_data = [
        # Pricing FAQs
        {
            "question": "What are your pricing plans?",
            "answer": "We offer three pricing plans: Starter ($49/month), Professional ($149/month), and Enterprise (custom pricing). Each plan includes different features and support levels.",
            "category": "pricing",
            "keywords": "pricing, plans, cost, price, rates, fees"
        },
        {
            "question": "Do you offer a free trial?",
            "answer": "Yes! We offer a 14-day free trial for all our plans. No credit card required to get started.",
            "category": "pricing",
            "keywords": "free trial, trial, demo, test, free"
        },
        {
            "question": "Can I change my plan later?",
            "answer": "Absolutely! You can upgrade or downgrade your plan at any time. Changes take effect immediately, and billing is prorated.",
            "category": "pricing",
            "keywords": "change plan, upgrade, downgrade, switch, modify"
        },
        
        # Features FAQs
        {
            "question": "What features are included in the platform?",
            "answer": "Our platform includes AI-powered lead scoring, automated content generation, social media scheduling, chatbot integration, and comprehensive analytics.",
            "category": "features",
            "keywords": "features, capabilities, functionality, tools, what included"
        },
        {
            "question": "How does the lead scoring work?",
            "answer": "Our AI analyzes visitor behavior, engagement patterns, and demographic data to score leads from 1-100. Higher scores indicate higher purchase intent.",
            "category": "features",
            "keywords": "lead scoring, AI scoring, lead qualification, lead rating"
        },
        {
            "question": "Can you integrate with our existing CRM?",
            "answer": "Yes! We integrate with popular CRMs including Salesforce, HubSpot, Pipedrive, and Zoho. Custom integrations are available for Enterprise plans.",
            "category": "features",
            "keywords": "integration, CRM, Salesforce, HubSpot, connect, sync"
        },
        
        # Support FAQs
        {
            "question": "What support options are available?",
            "answer": "We offer email support (24h response), live chat (business hours), and phone support for Enterprise customers. All plans include access to our knowledge base.",
            "category": "support",
            "keywords": "support, help, assistance, customer service, contact"
        },
        {
            "question": "Do you provide training?",
            "answer": "Yes! We offer onboarding sessions for new customers, video tutorials, and personalized training for Enterprise plans.",
            "category": "support",
            "keywords": "training, onboarding, tutorials, learning, education"
        },
        {
            "question": "What are your support hours?",
            "answer": "Email support is available 24/7. Live chat is available Monday-Friday 9 AM - 6 PM EST. Phone support (Enterprise) is available during business hours.",
            "category": "support",
            "keywords": "support hours, availability, when, time, schedule"
        },
        
        # Technical FAQs
        {
            "question": "Is the platform secure?",
            "answer": "Yes! We use enterprise-grade security including SSL encryption, SOC 2 compliance, and regular security audits. Your data is protected with industry-standard measures.",
            "category": "technical",
            "keywords": "security, secure, encryption, compliance, data protection"
        },
        {
            "question": "What browsers are supported?",
            "answer": "Our platform works with all modern browsers: Chrome, Firefox, Safari, and Edge. We recommend keeping your browser updated for the best experience.",
            "category": "technical",
            "keywords": "browsers, compatibility, Chrome, Firefox, Safari, Edge"
        }
    ]
    
    try:
        print("🌱 Seeding FAQ data...")
        
        for faq_data in faqs_data:
            # Check if FAQ already exists
            existing = db.query(faq_crud.model).filter(
                faq_crud.model.question == faq_data["question"]
            ).first()
            
            if not existing:
                faq_create = FAQCreate(**faq_data)
                faq_crud.create(db, obj_in=faq_create)
                print(f"   ✅ Added: {faq_data['question'][:50]}...")
            else:
                print(f"   ⏭️ Skipped (exists): {faq_data['question'][:50]}...")
        
        print(f"\n✅ Successfully seeded {len(faqs_data)} FAQs")
        
    except Exception as e:
        print(f"❌ Error seeding FAQs: {str(e)}")
    
    finally:
        db.close()


def main():
    """Main seeding function"""
    print("🚀 Chatbot Data Seeding")
    print("=" * 25)
    
    seed_faqs()
    
    print("\n🎉 Seeding completed!")


if __name__ == "__main__":
    main()
