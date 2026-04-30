from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # AI Provider: openai | groq | gemini
    ai_provider: str = Field("groq", env="AI_PROVIDER")

    # OpenAI
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")

    # Groq (free)
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # Google Gemini (free)
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", env="GEMINI_MODEL")

    # Telegram
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    telegram_allowed_users: str = Field("", env="TELEGRAM_ALLOWED_USERS")

    # FastAPI
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_base_url: str = Field("http://localhost:8000", env="API_BASE_URL")

    # MetaTrader 5
    mt5_enabled: bool = Field(False, env="MT5_ENABLED")
    mt5_login: int = Field(0, env="MT5_LOGIN")
    mt5_password: str = Field("", env="MT5_PASSWORD")
    mt5_server: str = Field("", env="MT5_SERVER")
    mt5_path: str = Field("", env="MT5_PATH")

    # Risk Management
    max_risk_percent: float = Field(1.0, env="MAX_RISK_PERCENT")
    max_spread_points: int = Field(30, env="MAX_SPREAD_POINTS")
    min_rr_ratio: float = Field(2.0, env="MIN_RR_RATIO")

    # Trading
    symbol: str = Field("XAUUSD", env="SYMBOL")
    auto_trade: bool = Field(False, env="AUTO_TRADE")

    # SMC Engine
    use_external_smc: bool = Field(False, env="USE_EXTERNAL_SMC")

    # Scanner / Auto-Alert
    scanner_enabled: bool = Field(True, env="SCANNER_ENABLED")
    scanner_interval_minutes: int = Field(15, env="SCANNER_INTERVAL_MINUTES")
    scanner_timeframes: str = Field("M15,H1,H4", env="SCANNER_TIMEFRAMES")

    # Morning Briefing
    briefing_hour: int = Field(7, env="BRIEFING_HOUR")
    briefing_minute: int = Field(0, env="BRIEFING_MINUTE")

    # Twelve Data (optional secondary data source)
    twelvedata_api_key: str = Field("", env="TWELVEDATA_API_KEY")

    # Default account balance used for risk sizing when user hasn't set one
    default_account_balance: float = Field(10000.0, env="DEFAULT_ACCOUNT_BALANCE")

    @property
    def scanner_timeframe_list(self) -> list[str]:
        return [tf.strip().upper() for tf in self.scanner_timeframes.split(",") if tf.strip()]

    @property
    def allowed_user_ids(self) -> list[int]:
        if not self.telegram_allowed_users:
            return []
        return [int(uid.strip()) for uid in self.telegram_allowed_users.split(",") if uid.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
