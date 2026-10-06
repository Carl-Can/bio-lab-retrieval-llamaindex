import os
import logging

# 让 bge-m3 在 GPU 上的前向可复现（cuBLAS 默认非确定，会让评测结果在边界 query 上漂移）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # API配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    API_KEY = os.getenv("API_KEY", "")

    # LLM配置
    # LLM_PROVIDER: openai | vllm（本地 vLLM OpenAI 兼容服务）
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # 本地 vLLM（OpenAI 兼容）
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen3.5-9B-AWQ")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
    VLLM_CONTEXT_WINDOW = int(os.getenv("VLLM_CONTEXT_WINDOW", 65536))
    VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", 1024))
    # Qwen3.5 是推理模型，RAG 合成时关掉思考更直接、更快
    VLLM_DISABLE_THINKING = os.getenv("VLLM_DISABLE_THINKING", "True").lower() == "true"

    # Embedding配置
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-m3")
    HF_DEVICE = os.getenv("HF_DEVICE", "cpu")
    # True: 只用本地缓存（离线）；False: 允许联网下载模型
    HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "True").lower() == "true"

    # 向量数据库配置
    VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
    # docstore / index_store 持久化目录（BM25、查看、删除需要文本与关系）
    STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")

    # 文档处理配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # 检索配置
    MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", 0.4))
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
    # RETRIEVAL_MODE: dense | exact（精确余弦,评测用） | hybrid（BM25 + 稠密 RRF，质量档）
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", 20))
    RRF_K = int(os.getenv("RRF_K", 60))

    # 重排配置
    # RERANK_PROVIDER: none | bge | bge-onnx | jev（质量档默认 bge）
    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "bge")
    RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    # ONNX int8 版（CPU/AVX2，比 PyTorch CPU 快）；可用 HF repo id 或本地缓存目录
    RERANK_ONNX_MODEL = os.getenv("RERANK_ONNX_MODEL", "kftof/bge-reranker-v2-m3-onnx-int8-avx2")
    RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", 512))
    # 送入重排的最大候选数（hybrid 融合后可能到 40，CPU 重排逐条打分，需设上限）
    RERANK_MAX_CANDIDATES = int(os.getenv("RERANK_MAX_CANDIDATES", 20))
    # 级联（provider=cascade）：Jev 粗筛 -> bge 精排
    RERANK_CASCADE_GATE = os.getenv("RERANK_CASCADE_GATE", "jev-gate")
    RERANK_CASCADE_FINE = os.getenv("RERANK_CASCADE_FINE", "bge")
    RERANK_CASCADE_GATE_TOP_K = int(os.getenv("RERANK_CASCADE_GATE_TOP_K", 6))
    # 高召回噪声门控（provider=jev-gate）：低阈值硬截断 + min_keep 兜底，保留原序
    JEV_GATE_THRESHOLD = float(os.getenv("JEV_GATE_THRESHOLD", 0.30))
    JEV_GATE_RELATIVE = float(os.getenv("JEV_GATE_RELATIVE", 0.0))
    JEV_GATE_MIN_KEEP = int(os.getenv("JEV_GATE_MIN_KEEP", 12))
    # 重排设备（仅 bge PyTorch 版用）；默认跟随 HF_DEVICE。vLLM 占满 GPU 时设为 cpu
    RERANK_DEVICE = os.getenv("RERANK_DEVICE", os.getenv("HF_DEVICE", "cpu"))
    RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", 20))
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))
    # Jev 决策模型（MiniCPM5-1B-AWQ-INT4，需 transformers>=5）
    JEV_MODEL = os.getenv("JEV_MODEL", "cyankiwi/MiniCPM5-1B-AWQ-INT4")
    JEV_LOCAL_PATH = os.getenv("JEV_LOCAL_PATH", "")
    JEV_DEVICE = os.getenv("JEV_DEVICE", "cuda")
    JEV_THRESHOLD = float(os.getenv("JEV_THRESHOLD", 0.45))
    JEV_MAX_LENGTH = int(os.getenv("JEV_MAX_LENGTH", 512))
    JEV_BATCH_SIZE = int(os.getenv("JEV_BATCH_SIZE", 8))

    # grounding / 幻觉核验
    # GROUNDING_PROVIDER: llm（LLM-as-judge，可靠） | jev（实验，1B NLI 实测不可用）
    GROUNDING_PROVIDER = os.getenv("GROUNDING_PROVIDER", "llm")
    GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", 0.5))
    GROUNDING_MAX_CLAIMS = int(os.getenv("GROUNDING_MAX_CLAIMS", 20))
    GROUNDING_MAX_CONTEXT_CHARS = int(os.getenv("GROUNDING_MAX_CONTEXT_CHARS", 4000))

    # 上传文件配置
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
    DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB

    # 其他配置
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    # 查询日志（jsonl），用于后续从真实使用中挖评测 query
    QUERY_LOG = os.getenv("QUERY_LOG", "./logs/queries.jsonl")

    @classmethod
    def setup_logging(cls):
        level = logging.DEBUG if cls.DEBUG else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
