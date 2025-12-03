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
    DATA_STORE_ID: str = os.getenv('DATA_STORE_ID', 'earnings-call-datastore')
    
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
    AGENT_TEMPERATURE: float = float(os.getenv('AGENT_TEMPERATURE', '0.7'))
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
  Data Store: {cls.DATA_STORE_ID}

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
