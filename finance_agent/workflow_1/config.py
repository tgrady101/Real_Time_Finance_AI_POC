"""Configuration management for the finance chatbot.

Centralizes all configuration including API keys, model settings,
and application parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


class Config:
    """Application configuration."""
    
    # API Keys
    FRED_API_KEY: str = os.getenv('FRED_API_KEY', '')
    ARIZE_SPACE_ID: str = os.getenv('ARIZE_SPACE_ID', '')
    ARIZE_API_KEY: str = os.getenv('ARIZE_API_KEY', '')
    
    # Google Cloud Configuration
    GOOGLE_CLOUD_PROJECT: str = os.getenv('GOOGLE_CLOUD_PROJECT', '')
    # Note: gemini-3-pro-preview only supports 'global' region
    GOOGLE_CLOUD_LOCATION: str = os.getenv('GOOGLE_CLOUD_LOCATION', 'global')
    
    # Vertex AI Vector Search Configuration
    VECTOR_SEARCH_LOCATION: str = os.getenv('VECTOR_SEARCH_LOCATION', 'us-central1')
    VECTOR_SEARCH_INDEX_ENDPOINT_ID: str = os.getenv('VECTOR_SEARCH_INDEX_ENDPOINT_ID', '')
    VECTOR_SEARCH_DEPLOYED_INDEX_ID: str = os.getenv('VECTOR_SEARCH_DEPLOYED_INDEX_ID', 'earnings_hybrid_3072')
    
    # Embedding Configuration (for manual embeddings)
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')
    EMBEDDING_LOCATION: str = os.getenv('EMBEDDING_LOCATION', 'us-central1')  # Embeddings work best here
    # 3072 is the ONLY dimension that is pre-normalized by Vertex AI
    # Lower dimensions (768, 1536) require manual L2 normalization for accurate similarity
    EMBEDDING_DIMENSIONALITY: int = int(os.getenv('EMBEDDING_DIMENSIONALITY', '3072'))
    VECTOR_STORE_PATH: str = os.getenv('VECTOR_STORE_PATH', 'data/earnings_vectors.json')
    
    # Model Configuration
    # Default model for general use
    # gemini-3-pro-preview requires enabling in Vertex AI Model Garden first
    # Fallback: gemini-2.5-flash works without additional setup
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-3-pro-preview')
    
    # Model tiers for dynamic routing (via before_model_callback)
    # Complex: For detailed analysis, multi-step reasoning, complex financial queries
    MODEL_COMPLEX: str = os.getenv('MODEL_COMPLEX', 'gemini-3-pro-preview')
    # Fast: For simple lookups, validation, basic queries (lower cost, faster response)
    MODEL_FAST: str = os.getenv('MODEL_FAST', 'gemini-2.5-flash')
    
    # Agent Parameters
    # Temperature fixed at 1.0 - do not change (optimal for Gemini models)
    AGENT_TEMPERATURE: float = 1.0
    AGENT_MAX_TOKENS: int = int(os.getenv('AGENT_MAX_TOKENS', '2048'))
    
    # Rate Limiting
    FRED_API_RATE_LIMIT: float = float(os.getenv('FRED_API_RATE_LIMIT', '0.5'))
    
    # S&P 500 Configuration
    SP500_REFRESH_DAYS: int = int(os.getenv('SP500_REFRESH_DAYS', '7'))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError if required keys are missing
        """
        required_keys = [
            ('GOOGLE_CLOUD_PROJECT', cls.GOOGLE_CLOUD_PROJECT),
        ]
        
        missing = [name for name, value in required_keys if not value]
        
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}. "
                f"Please set these in your .env file."
            )
        
        return True
    
    @classmethod
    def get_summary(cls) -> str:
        """Get a summary of the current configuration (masking sensitive values).
        
        Returns:
            Configuration summary string
        """
        def mask_key(key: str) -> str:
            if not key:
                return "[NOT SET]"
            if len(key) <= 10:
                return f"{key[:4]}****"
            return f"{key[:10]}****"
        
        return f"""
Configuration Summary:
======================
API Keys:
  FRED API: {mask_key(cls.FRED_API_KEY)}
  Arize: {mask_key(cls.ARIZE_API_KEY)}

Google Cloud:
  Project: {cls.GOOGLE_CLOUD_PROJECT}
  Location: {cls.GOOGLE_CLOUD_LOCATION}
  Vector Search Location: {cls.VECTOR_SEARCH_LOCATION}
  Vector Search Endpoint: {cls.VECTOR_SEARCH_INDEX_ENDPOINT_ID or '[NOT SET]'}

Models:
  Default: {cls.GEMINI_MODEL}
  Complex (analysis): {cls.MODEL_COMPLEX}
  Fast (simple queries): {cls.MODEL_FAST}
  Temperature: {cls.AGENT_TEMPERATURE}
  Max Tokens: {cls.AGENT_MAX_TOKENS}
"""


# Singleton config instance
config = Config()


if __name__ == "__main__":
    print(config.get_summary())
    
    try:
        config.validate()
        print("\n✓ Configuration is valid")
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
