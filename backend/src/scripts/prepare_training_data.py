import argparse
import pandas as pd
import json
import os
import random
import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict

def create_diverse_content_templates() -> Dict[str, Dict[str, List[str]]]:
    """Create diverse content templates for different tones and channels"""
    return {
        "Professional": {
            "Social Media": [
                "We're excited to announce {product}, specifically designed for {audience} in the {industry} sector. This innovative solution {benefit} while maintaining the highest industry standards. Learn more about how we can help transform your business.",
                "Introducing {product} - the comprehensive solution that {audience} have been waiting for. Our latest innovation {benefit} and is backed by our deep expertise in {industry}. Contact us to discover the difference.",
                "{product} represents the next evolution in {industry} solutions. Designed with {audience} in mind, it {benefit} through cutting-edge technology and proven methodologies."
            ],
            "Email": [
                "Dear {audience},\n\nWe are pleased to introduce {product}, our latest solution designed specifically for professionals in the {industry} industry. This innovative platform {benefit} while ensuring the highest standards of security and reliability.\n\nKey features include:\n• Advanced analytics and reporting\n• Seamless integration with existing systems\n• 24/7 enterprise support\n\nWe would be delighted to schedule a demonstration at your convenience.",
                "Subject: Introducing {product} - Transforming {industry} Operations\n\nHello,\n\nAs a leader in {industry}, you understand the importance of staying ahead. That's why we've developed {product} to help {audience} like you {benefit} more effectively.\n\nOur solution offers:\n• Proven ROI within 90 days\n• Enterprise-grade security\n• Comprehensive training and support\n\nLet's discuss how {product} can drive your success forward."
            ],
            "Product Description": [
                "{product} is an enterprise-grade solution engineered for {audience} in the {industry} sector. This comprehensive platform {benefit} through advanced algorithms and intuitive design. Built with scalability and security in mind, {product} integrates seamlessly with existing workflows while providing powerful analytics and reporting capabilities.",
                "Transform your {industry} operations with {product}, the professional solution designed for {audience}. Our platform {benefit} through innovative technology and proven methodologies. With enterprise-grade security, comprehensive support, and seamless integration capabilities, {product} delivers measurable results from day one."
            ],
            "Ad": [
                "Discover {product} - the professional solution that's revolutionizing how {audience} work in {industry}. Our platform {benefit} while maintaining the highest standards of security and compliance. Join thousands of satisfied customers who have transformed their operations. Request a demo today.",
                "Ready to elevate your {industry} performance? {product} is the comprehensive solution designed for forward-thinking {audience}. Experience how our platform {benefit} through proven technology and expert support. Schedule your consultation now."
            ]
        },
        "Casual": {
            "Social Media": [
                "Hey {audience}! 👋 Just dropped something awesome for the {industry} community - meet {product}! This little game-changer helps you {benefit} without all the usual headaches. Perfect for busy professionals who want results, not complications! ✨",
                "Exciting news, {audience}! 🎉 We've been working on something special for {industry} folks, and it's finally here - {product}! It's designed to help you {benefit} in the smartest way possible. Check it out and let us know what you think! 🚀",
                "Calling all {audience} in {industry}! 📢 Say hello to {product} - your new favorite tool that actually makes work fun again. It {benefit} while keeping things simple and straightforward. Trust us, you're going to love this! 💪"
            ],
            "Email": [
                "Hey there!\n\nHope you're having a great day! We wanted to share something pretty cool with you - {product}, our latest creation for awesome {audience} like yourself.\n\nHere's the deal: we know {industry} can be tough, so we built something that {benefit} without making your life more complicated. It's actually pretty fun to use (we promise!).\n\nWant to take it for a spin? We'd love to show you around! 😊",
                "Hi {audience}!\n\nGot a minute? We've been cooking up something special for the {industry} community, and we think you're going to dig it.\n\n{product} is our answer to all those times you wished there was an easier way to {benefit}. No more jumping through hoops or dealing with overly complicated tools.\n\nInterested? Let's chat! We're always happy to talk shop with fellow {industry} enthusiasts. 🤝"
            ],
            "Product Description": [
                "Meet {product} - the friendly solution that gets {audience} in {industry}! We built this because we know you're busy and don't have time for complicated tools that promise the world but deliver headaches. {product} actually {benefit} in a way that makes sense, without all the unnecessary bells and whistles. It's intuitive, reliable, and honestly pretty fun to use.",
                "{product} is what happens when you combine great technology with a deep understanding of what {audience} really need. No corporate jargon, no overly complex features - just a solid tool that {benefit} exactly the way you'd want it to. Built by {industry} people, for {industry} people."
            ],
            "Ad": [
                "Tired of tools that promise everything but deliver headaches? {product} is different. Built specifically for {audience} who want to {benefit} without the drama. Simple, effective, and actually enjoyable to use. Give it a try - we think you'll love it! 🎯",
                "Hey {audience}! Ready for something that actually works? {product} is the no-nonsense solution for {industry} professionals who want to {benefit} without jumping through hoops. Less complexity, more results. Sounds good? Let's talk! 🚀"
            ]
        },
        "Witty": {
            "Social Media": [
                "Plot twist: {product} just made {benefit} actually fun for {audience}! 🎭 Who knew {industry} could be this entertaining? Say goodbye to boring solutions and hello to something that actually sparks joy (yes, we went there). Ready to shake things up? 🚀",
                "Breaking news: {product} is here to save {audience} from the land of boring {industry} tools! ⚡ Finally, a solution that {benefit} AND doesn't put you to sleep. We're basically the superhero of software (cape not included, but attitude definitely is). 🦸‍♂️",
                "Alert: {product} has entered the chat and it's about to change everything for {audience} in {industry}! 🔥 Tired of tools that make you want to bang your head against the wall? This one actually {benefit} AND makes you smile. Revolutionary, right? 😎"
            ],
            "Email": [
                "Subject: Finally! A {industry} tool that doesn't hate you back\n\nHey {audience},\n\nWe have a confession: we were tired of watching brilliant people like you struggle with terrible {industry} tools. So we did something about it.\n\nMeet {product} - the solution that {benefit} without making you question your life choices. It's like having a really smart friend who actually knows what they're doing (and doesn't judge you for asking questions).\n\nWant to see how we're making {industry} fun again? Let's chat! ⚡",
                "Subject: {product} - Because life's too short for boring {industry} tools\n\nHello brilliant {audience}!\n\nEver wonder why {industry} solutions are designed by people who clearly hate joy? We wondered the same thing, so we built {product} to prove that you can {benefit} AND have fun doing it.\n\nThink of us as the anti-boring solution. We're here to make your {industry} life better, not more complicated.\n\nCurious? We'd love to show you how we're shaking things up! 🚀"
            ],
            "Product Description": [
                "Meet {product} - the {industry} solution that dared to ask 'what if software didn't suck?' Built for {audience} who are tired of tools that seem designed by people who hate their users, {product} actually {benefit} in ways that make sense. We've eliminated the unnecessary complexity, corporate speak, and soul-crushing user experiences that plague our industry. The result? A tool that's powerful, intuitive, and (dare we say it) actually enjoyable to use.",
                "{product} is what happens when you combine serious {industry} expertise with a healthy disrespect for 'that's how we've always done it.' Designed for {audience} who want to {benefit} without sacrificing their sanity, our platform proves that professional tools don't have to be punishing. We've packed enterprise-grade capabilities into an experience that's actually human-friendly. Revolutionary? Maybe. Overdue? Definitely."
            ],
            "Ad": [
                "Attention {audience}: {product} is here to rescue you from the wasteland of terrible {industry} tools! 🦸‍♀️ Finally, a solution that {benefit} without making you want to throw your computer out the window. We're basically the fun police, but for boring software. Ready to join the revolution? 🚀",
                "Breaking: Local {industry} tool doesn't make users cry! {product} is the plot twist your workflow has been waiting for. Built for {audience} who believe that {benefit} shouldn't require a degree in frustration management. Come for the features, stay for the sanity. 🎯"
            ]
        }
    }

def get_enhanced_benefits_map() -> Dict[str, List[str]]:
    """Enhanced benefits mapping for different products"""
    return {
        "CRM Tool": [
            "streamline customer relationships and boost sales efficiency",
            "transform chaotic customer data into actionable insights",
            "automate repetitive sales tasks and focus on closing deals",
            "create personalized customer experiences at scale",
            "track customer interactions and never miss a follow-up"
        ],
        "AI Assistant": [
            "automate routine tasks and free up time for strategic work",
            "provide intelligent insights that drive better decision-making",
            "streamline workflows and eliminate manual bottlenecks",
            "enhance productivity through smart automation",
            "deliver personalized support and guidance 24/7"
        ],
        "Math Tutor App": [
            "accelerate learning with personalized instruction",
            "build mathematical confidence through adaptive practice",
            "provide step-by-step guidance for complex problems",
            "track progress and identify areas for improvement",
            "make math engaging and accessible for all learning styles"
        ],
        "Cloud Storage": [
            "secure your data with enterprise-grade encryption",
            "enable seamless collaboration across teams and locations",
            "provide instant access to files from any device",
            "automatically backup and sync your important documents",
            "scale storage capacity as your business grows"
        ],
        "Yoga Mat": [
            "provide superior comfort and stability during practice",
            "support proper alignment with innovative design features",
            "offer durability that withstands intensive daily use",
            "enhance your yoga experience with premium materials",
            "deliver the perfect balance of grip and cushioning"
        ],
        "Protein Bar": [
            "fuel your body with clean, high-quality nutrition",
            "provide sustained energy for your active lifestyle",
            "support muscle recovery and growth with complete proteins",
            "satisfy cravings with delicious, guilt-free nutrition",
            "deliver convenient nutrition that fits your busy schedule"
        ]
    }

def get_audience_variations() -> Dict[str, List[str]]:
    """Get variations for different audience types"""
    return {
        "professionals": ["business professionals", "industry experts", "experienced practitioners", "seasoned professionals"],
        "students": ["learners", "aspiring professionals", "academic achievers", "educational enthusiasts"],
        "small businesses": ["growing companies", "entrepreneurial ventures", "small business owners", "startup teams"],
        "startups": ["innovative companies", "emerging businesses", "entrepreneurial teams", "growth-stage companies"],
        "remote teams": ["distributed teams", "virtual workforces", "remote professionals", "digital nomads"],
        "fitness lovers": ["health enthusiasts", "active individuals", "wellness seekers", "fitness professionals"]
    }

def enhance_dataset_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Create more diverse and realistic training examples"""
    templates = create_diverse_content_templates()
    benefits_map = get_enhanced_benefits_map()
    audience_variations = get_audience_variations()
    
    enhanced_examples = []
    
    for _, row in df.iterrows():
        # Extract basic information
        original_prompt = row.get("Prompt", "")
        tone = row.get("Tone", "Professional")
        channel = row.get("Channel", "Social Media")
        industry = row.get("Industry", "Tech")
        
        # Extract product from prompt or use default
        product = extract_product_from_prompt(original_prompt)
        if not product:
            product = "our solution"
        
        # Get audience from prompt or use default
        audience = extract_audience_from_prompt(original_prompt)
        if not audience:
            audience = "professionals"
        
        # Get variations for audience
        audience_vars = audience_variations.get(audience.lower(), [audience])
        selected_audience = random.choice([audience] + audience_vars)
        
        # Get appropriate templates and benefits
        tone_templates = templates.get(tone, templates["Professional"])
        channel_templates = tone_templates.get(channel, tone_templates.get("Social Media", []))
        
        if not channel_templates:
            # Fallback to any available template for this tone
            channel_templates = next(iter(tone_templates.values()))
        
        # Select template and benefit
        template = random.choice(channel_templates)
        benefits = benefits_map.get(product, ["deliver exceptional value and results"])
        benefit = random.choice(benefits)
        
        # Generate enhanced content
        try:
            enhanced_content = template.format(
                product=product,
                audience=selected_audience,
                benefit=benefit,
                industry=industry
            )
        except KeyError as e:
            # Fallback if formatting fails
            enhanced_content = f"Introducing {product} for {selected_audience} in {industry}. This solution helps you {benefit}."
        
        # Create enhanced example
        enhanced_examples.append({
            "Prompt": original_prompt,
            "Tone": tone,
            "Channel": channel,
            "Industry": industry,
            "Generated_Content": enhanced_content,
            "Product": product,
            "Audience": selected_audience,
            "Benefit": benefit
        })
    
    return pd.DataFrame(enhanced_examples)

def extract_product_from_prompt(prompt: str) -> str:
    """Extract product name from prompt"""
    # Common product patterns
    product_patterns = [
        r"(?:our|the|new)\s+([A-Z][A-Za-z\s]+(?:App|Tool|Mat|Bar|Storage|Assistant))",
        r"([A-Z][A-Za-z\s]+(?:App|Tool|Mat|Bar|Storage|Assistant))",
        r"(?:about|for|promoting)\s+([A-Z][A-Za-z\s]+)"
    ]
    
    for pattern in product_patterns:
        match = re.search(pattern, prompt)
        if match:
            product = match.group(1).strip()
            # Clean up common words
            product = re.sub(r'^(our|the|new)\s+', '', product, flags=re.IGNORECASE)
            return product
    
    return None

def extract_audience_from_prompt(prompt: str) -> str:
    """Extract target audience from prompt"""
    audience_patterns = [
        r"for\s+(students|professionals|startups|small businesses|remote teams|fitness lovers)",
        r"to\s+(students|professionals|startups|small businesses|remote teams|fitness lovers)",
        r"(students|professionals|startups|small businesses|remote teams|fitness lovers)"
    ]
    
    for pattern in audience_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return "professionals"

def validate_dataset_quality(df: pd.DataFrame) -> Dict[str, any]:
    """Validate and analyze dataset quality"""
    quality_report = {
        "total_samples": len(df),
        "unique_prompts": df['Prompt'].nunique(),
        "unique_content": df['Generated_Content'].nunique(),
        "tone_distribution": df['Tone'].value_counts().to_dict(),
        "channel_distribution": df['Channel'].value_counts().to_dict(),
        "industry_distribution": df['Industry'].value_counts().to_dict(),
        "avg_content_length": df['Generated_Content'].str.len().mean(),
        "content_length_std": df['Generated_Content'].str.len().std(),
        "issues": []
    }
    
    # Check for issues
    if quality_report["unique_content"] / quality_report["total_samples"] < 0.8:
        quality_report["issues"].append("Low content diversity - many repeated outputs")
    
    if quality_report["avg_content_length"] < 50:
        quality_report["issues"].append("Content appears too short on average")
    
    if quality_report["content_length_std"] < 20:
        quality_report["issues"].append("Content length variation is very low")
    
    # Check for template-like content
    template_indicators = ["Introducing our latest solution", "🎉 Have you seen our new"]
    template_count = sum(df['Generated_Content'].str.contains(indicator, case=False, regex=False).sum() 
                        for indicator in template_indicators)
    
    if template_count > len(df) * 0.3:
        quality_report["issues"].append("High template content detected")
    
    return quality_report

def create_evaluation_split(df: pd.DataFrame, eval_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create training and evaluation splits"""
    # Ensure we have examples from each tone-channel combination
    stratified_samples = []
    remaining_samples = []
    
    # Group by tone and channel
    groups = df.groupby(['Tone', 'Channel'])
    
    for (tone, channel), group in groups:
        if len(group) >= 10:  # Ensure enough samples for meaningful split
            eval_size = max(1, int(len(group) * eval_ratio))
            eval_samples = group.sample(n=eval_size, random_state=42)
            train_samples = group.drop(eval_samples.index)
            
            stratified_samples.append(eval_samples)
            remaining_samples.append(train_samples)
        else:
            # If too few samples, keep all for training
            remaining_samples.append(group)
    
    # Combine results
    eval_df = pd.concat(stratified_samples, ignore_index=True) if stratified_samples else pd.DataFrame()
    train_df = pd.concat(remaining_samples, ignore_index=True)
    
    return train_df, eval_df

def process_training_dataset(input_file: str, output_file: str, enhance_diversity: bool = True) -> None:
    """Process the training dataset to format required for fine-tuning."""
    print(f"Processing dataset from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check required columns
    required_columns = ["Prompt", "Tone", "Channel", "Industry", "Generated_Content"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    print(f"Original dataset: {len(df)} samples")
    
    # Enhance diversity if requested
    if enhance_diversity:
        print("Enhancing dataset diversity...")
        df = enhance_dataset_diversity(df)
        print(f"Enhanced dataset: {len(df)} samples")
    
    # Validate dataset quality
    quality_report = validate_dataset_quality(df)
    print(f"Dataset Quality Report:")
    print(f"  - Total samples: {quality_report['total_samples']}")
    print(f"  - Unique content ratio: {quality_report['unique_content']/quality_report['total_samples']:.2%}")
    print(f"  - Average content length: {quality_report['avg_content_length']:.1f} characters")
    
    if quality_report["issues"]:
        print("  - Quality Issues:")
        for issue in quality_report["issues"]:
            print(f"    * {issue}")
    
    # Create train/eval split
    train_df, eval_df = create_evaluation_split(df)
    print(f"Split: {len(train_df)} training, {len(eval_df)} evaluation samples")
    
    # Transform data into format for fine-tuning
    def create_processed_data(source_df, prefix=""):
        processed_data = []
        for _, row in source_df.iterrows():
            # Create metadata
            metadata = {
                "tone": row["Tone"],
                "channel": row["Channel"],
                "industry": row["Industry"],
                "product": row.get("Product", ""),
                "audience": row.get("Audience", ""),
                "benefit": row.get("Benefit", "")
            }
            
            # Create training example
            processed_data.append({
                "prompt": row["Prompt"],
                "content": row["Generated_Content"],
                "metadata": json.dumps(metadata)
            })
        return processed_data
    
    # Process training data
    processed_train_data = create_processed_data(train_df)
    train_df_processed = pd.DataFrame(processed_train_data)
    
    # Save training data
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_df_processed.to_csv(output_file, index=False)
    print(f"Training data saved to {output_file}")
    
    # Save evaluation data if it exists
    if len(eval_df) > 0:
        eval_output_file = output_file.replace('.csv', '_eval.csv')
        processed_eval_data = create_processed_data(eval_df)
        eval_df_processed = pd.DataFrame(processed_eval_data)
        eval_df_processed.to_csv(eval_output_file, index=False)
        print(f"Evaluation data saved to {eval_output_file}")
    
    # Save quality report
    quality_report_file = output_file.replace('.csv', '_quality_report.json')
    with open(quality_report_file, 'w') as f:
        json.dump(quality_report, f, indent=2)
    print(f"Quality report saved to {quality_report_file}")
    
    print(f"Dataset processing completed successfully!")

def main():
    """Main function to prepare training data."""
    parser = argparse.ArgumentParser(description="Prepare training data for content generation model")
    parser.add_argument("--input", required=True, help="Path to input CSV file with content data")
    parser.add_argument("--output", required=True, help="Path to output CSV file for training")
    parser.add_argument("--no-enhance", action="store_true", help="Skip diversity enhancement")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Ratio of data to use for evaluation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    enhance_diversity = not args.no_enhance
    process_training_dataset(args.input, args.output, enhance_diversity)

if __name__ == "__main__":
    main()