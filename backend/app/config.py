"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), "../../.env")

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""

    # Flask配置
    SECRET_KEY = os.environ.get("SECRET_KEY", "mirofish-secret-key")
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False

    # ===========================================
    # LLM配置（远端 API - 阿里百炼/OpenAI）
    # ===========================================
    LLM_API_KEY = os.environ.get("LLM_API_KEY")
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

    # ===========================================
    # Embedding配置（本地 vllm conda 环境）
    # ===========================================
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "/home/ying/Qwen3-Embedding-4B")
    EMBEDDING_BASE_URL = os.environ.get(
        "EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"
    )
    # Embedding API Key（vllm 不验证时可用 "EMPTY"）
    EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "EMPTY")

    # ===========================================
    # Neo4j配置（本地自托管）
    # ===========================================
    NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://127.0.0.1:7687")
    NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

    # ===========================================
    # Zep配置（旧版，将逐步弃用）
    # ===========================================
    ZEP_API_KEY = os.environ.get("ZEP_API_KEY")

    # ===========================================
    # 功能开关
    # ===========================================
    USE_MEM0 = os.environ.get("USE_MEM0", "true").lower() == "true"

    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../uploads")
    ALLOWED_EXTENSIONS = {"pdf", "md", "txt", "markdown"}

    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小

    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get("OASIS_DEFAULT_MAX_ROUNDS", "10"))
    OASIS_SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__), "../uploads/simulations"
    )

    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        "CREATE_POST",
        "LIKE_POST",
        "REPOST",
        "FOLLOW",
        "DO_NOTHING",
        "QUOTE_POST",
    ]
    OASIS_REDDIT_ACTIONS = [
        "LIKE_POST",
        "DISLIKE_POST",
        "CREATE_POST",
        "CREATE_COMMENT",
        "LIKE_COMMENT",
        "DISLIKE_COMMENT",
        "SEARCH_POSTS",
        "SEARCH_USER",
        "TREND",
        "REFRESH",
        "DO_NOTHING",
        "FOLLOW",
        "MUTE",
    ]

    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(
        os.environ.get("REPORT_AGENT_MAX_TOOL_CALLS", "5")
    )
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(
        os.environ.get("REPORT_AGENT_MAX_REFLECTION_ROUNDS", "2")
    )
    REPORT_AGENT_TEMPERATURE = float(os.environ.get("REPORT_AGENT_TEMPERATURE", "0.5"))

    @classmethod
    def validate(cls):
        """验证必要配置"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")

        # ZEP_API_KEY 改为可选（使用 mem0 时不需要）
        # if not cls.ZEP_API_KEY:
        #     errors.append("ZEP_API_KEY 未配置")

        # 验证 Neo4j 配置（使用 mem0 时需要）
        if cls.USE_MEM0:
            if not cls.NEO4J_PASSWORD:
                errors.append("NEO4J_PASSWORD 未配置（USE_MEM0=true 时必需）")

        return errors
