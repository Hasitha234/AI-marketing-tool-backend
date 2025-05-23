"""
Manual testing script for the chatbot
Run this to test your chatbot functionality
"""

import asyncio
import json
import os
import sys



from app.services.chatbot_service import ChatbotService, DialogflowService
from app.db.session import SessionLocal


async def test_chatbot_conversation():
    """Test a full conversation with the chatbot"""
    
    print("🤖 AI Marketing Chatbot Test")
    print("=" * 50)
    
    # Initialize services
    db = SessionLocal()
    chatbot_service = ChatbotService(db)
    session_id = "test-session-" + str(asyncio.get_running_loop().time())
    
    # Test messages
    test_messages = [
        "Hello",
        "Tell me about your pricing plans",
        "What features do you offer?",
        "I want to schedule a demo",
        "I need to contact sales",
        "I'm a small business looking for marketing automation"
    ]
    
    try:
        for i, message in enumerate(test_messages, 1):
            print(f"\n👤 User: {message}")
            
            # Process message
            response = chatbot_service.process_message(
                session_id=session_id,
                message=message,
                user_id=None
            )
            
            print(f"🤖 Bot: {response['response']}")
            print(f"📊 Intent: {response['intent']} (Confidence: {response['confidence']:.2f})")
            
            if response.get('quick_replies'):
                print(f"💡 Quick Replies: {', '.join(response['quick_replies'])}")
            
            if response.get('actions'):
                print(f"🎯 Actions: {', '.join(response['actions'])}")
            
            # Small delay between messages
            await asyncio.sleep(1)
        
        # Get conversation history
        print(f"\n📖 Conversation History:")
        print("=" * 30)
        
        history = chatbot_service.get_chat_history(session_id)
        for msg in history:
            speaker = "👤 User" if msg['is_user'] else "🤖 Bot"
            print(f"{speaker}: {msg['message']}")
        
        # Get session analytics
        print(f"\n📈 Session Analytics:")
        print("=" * 25)
        
        analytics = chatbot_service.get_session_analytics(session_id)
        if analytics:
            print(f"Total Messages: {analytics['total_messages']}")
            print(f"User Messages: {analytics['user_messages']}")
            print(f"Bot Messages: {analytics['bot_messages']}")
            print(f"Unique Intents: {len(analytics['unique_intents'])}")
            print(f"Average Confidence: {analytics['average_confidence']:.2f}")
            print(f"Session Duration: {analytics['session_duration']:.1f} minutes")
    
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
    
    finally:
        db.close()
        print(f"\n✅ Test completed for session: {session_id}")


def test_dialogflow_setup():
    """Test Dialogflow connection"""
    print("🔗 Testing Dialogflow Connection")
    print("=" * 35)
    
    try:
        dialogflow_service = DialogflowService()
        
        # Test basic intent detection
        test_phrases = [
            "Hello there!",
            "What are your pricing plans?",
            "I want to schedule a demo",
            "Tell me about features",
            "I need support"
        ]
        
        for phrase in test_phrases:
            print(f"\n📝 Testing: '{phrase}'")
            result = dialogflow_service.detect_intent("test-session", phrase)
            print(f"   Intent: {result['intent']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Response: {result['response_text'][:100]}...")
        
        print("\n✅ Dialogflow connection successful!")
        
    except Exception as e:
        print(f"❌ Dialogflow connection failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check GOOGLE_CLOUD_PROJECT_ID environment variable")
        print("2. Verify GOOGLE_APPLICATION_CREDENTIALS path")
        print("3. Ensure Dialogflow API is enabled")
        print("4. Check service account permissions")


async def interactive_test():
    """Interactive chatbot testing"""
    print("🎮 Interactive Chatbot Test")
    print("=" * 30)
    print("Type your messages (or 'quit' to exit)")
    
    db = SessionLocal()
    chatbot_service = ChatbotService(db)
    session_id = f"interactive-{asyncio.get_running_loop().time()}"
    
    try:
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process message
            response = chatbot_service.process_message(
                session_id=session_id,
                message=user_input,
                user_id=None
            )
            
            print(f"🤖 Bot: {response['response']}")
            
            # Show quick replies if available
            if response.get('quick_replies'):
                print("💡 Quick replies:")
                for i, reply in enumerate(response['quick_replies'], 1):
                    print(f"   {i}. {reply}")
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        db.close()


def main():
    """Main testing function"""
    print("🚀 Chatbot Testing Suite")
    print("=" * 25)
    
    choice = input("""
Choose test type:
1. Dialogflow Connection Test
2. Automated Conversation Test  
3. Interactive Test
4. All Tests

Enter choice (1-4): """).strip()
    
    if choice == "1":
        test_dialogflow_setup()
    elif choice == "2":
        asyncio.run(test_chatbot_conversation())
    elif choice == "3":
        asyncio.run(interactive_test())
    elif choice == "4":
        print("\n🔄 Running all tests...")
        test_dialogflow_setup()
        print("\n" + "="*50)
        asyncio.run(test_chatbot_conversation())
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
