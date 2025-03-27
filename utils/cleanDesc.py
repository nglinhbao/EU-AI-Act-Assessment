import re

def clean_description(description):
    """
    Clean and normalize AI application descriptions.
    
    Args:
        description (str): The original description text
        
    Returns:
        str: Cleaned description with only relevant English text
    """
    if not description or not isinstance(description, str):
        return ""
    
    # Remove URLs
    description = re.sub(r'https?://\S+', '', description)
    
    # Remove emoji codes like :robot:, :alarm_clock:, etc.
    description = re.sub(r':[a-z_]+:', '', description)
    
    # Remove emojis and special characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    description = emoji_pattern.sub(r'', description)
    
    # Try to find long English sections first (like complete sentences)
    english_sentences = re.findall(r'[A-Z][^.!?]*[.!?]', description)
    if english_sentences:
        description = ' '.join(english_sentences)
    else:
        # Check if we have any substantial English text (at least 20 chars with English words)
        english_match = re.search(r'[A-Za-z][A-Za-z\s.,;:!?&()+-]{20,}', description)
        if english_match:
            description = english_match.group(0)
        else:
            # Remove all non-ASCII characters and hope for the best
            description = re.sub(r'[^\x00-\x7F]+', ' ', description)
    
    # Explicitly check for broken descriptions with repeated 'ai' at the start
    description = re.sub(r'^(ai)+\s*', '', description, flags=re.IGNORECASE)
    
    # Remove social media handles and hashtags
    description = re.sub(r'[@#]\S+', '', description)
    
    # Keep ampersands and important special characters
    description = re.sub(r'[^\w\s.,;:!?()&+\-]', '', description)
    
    # Remove excess whitespace
    description = re.sub(r'\s+', ' ', description)
    
    # Remove trailing/leading citation marks and brackets
    description = re.sub(r'^\s*["\']|["\']\s*$', '', description)
    description = re.sub(r'^\s*\(|\)\s*$', '', description)
    
    # Fix spacing around punctuation
    description = re.sub(r'\s+([.,;:!?)])', r'\1', description)
    description = re.sub(r'([({])\s+', r'\1', description)
    
    # Remove references to specific platforms like GitHub, Discord, etc.
    description = re.sub(r'\s*\(?join here:.*?\)?', '', description)
    description = re.sub(r'\s*\(?on github\)?', '', description, flags=re.IGNORECASE)
    
    # Remove common prefixes and suffixes that don't add meaning
    description = re.sub(r'^\s*(this is|a|the)\s+', '', description, flags=re.IGNORECASE)
    
    # Fix P2P being split to P P
    description = re.sub(r'P\s+P', 'P2P', description)
    
    # Strip and return
    return description.strip()

if __name__ == "__main__":
    # Example usage
    examples = [
        "The AI Toolkit for TypeScript. From the creators of Next.js, the AI SDK is a free open-source library for building AI-powered applications and agents ",
        "🤖 可 DIY 的 多模态 AI 聊天机器人 | 🚀 快速接入 微信、 QQ、Telegram、等聊天平台 | 🦈支持DeepSeek、Grok、Claude、Ollama、Gemini、OpenAI | 工作流系统、网页搜索、AI画图、人设调教、虚拟女仆、语音对话 | ",
        "🚀🤖 Crawl4AI: Open-source LLM Friendly Web Crawler & Scraper. Don't be shy, join here: https://discord.gg/mEkkMXFG",
        "ai副业赚钱大集合，教你如何利用ai做一些副业项目，赚取更多额外收益。The Ultimate Guide to Making Money with AI Side Hustles: Learn how to leverage AI for some cool side gigs and rake in some extra cash. Check out the English version for more insights.",
        ":robot: The free, Open Source alternative to OpenAI, Claude and others. Self-hosted and local-first. Drop-in replacement for OpenAI,  running on consumer-grade hardware. No GPU required. Runs gguf, transformers, diffusers and many more models architectures. Features: Generate Text, Audio, Video, Images, Voice Cloning, Distributed, P2P inference"
    ]
    
    for example in examples:
        cleaned = clean_description(example)
        print(f"Original: {example}")
        print(f"Cleaned: {cleaned}")
        print("-" * 80)
