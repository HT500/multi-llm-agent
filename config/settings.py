"""設定ファイル - APIキー管理、閾値設定、モデル設定、ローカルLLM設定"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# キャッシュファイルパス
MODEL_CACHE_FILE = PROJECT_ROOT / '.model_cache.json'

# 基本設定
COVERAGE_THRESHOLD = float(os.getenv('COVERAGE_THRESHOLD', '0.8'))  # 網羅性閾値
MAX_RESEARCH_ITERATIONS = int(os.getenv('MAX_RESEARCH_ITERATIONS', '5'))  # 最大調査イテレーション数

# LLMモード: "api" または "local"
LLM_MODE = os.getenv('LLM_MODE', 'api')

# API LLM設定
LLM_CONFIG = {
    'openai': {
        'enabled': os.getenv('OPENAI_ENABLED', 'true').lower() == 'true',
        'api_key': os.getenv('OPENAI_API_KEY'),
        'models': os.getenv('OPENAI_MODELS', 'gpt-5.2,gpt-4,gpt-3.5-turbo').split(','),
        'auto_discover': os.getenv('OPENAI_AUTO_DISCOVER', 'false').lower() == 'true',
        'fallback_strategy': os.getenv('OPENAI_FALLBACK_STRATEGY', 'use_first_available'),
        'cache_models': os.getenv('OPENAI_CACHE_MODELS', 'true').lower() == 'true',
        'cache_duration_hours': int(os.getenv('OPENAI_CACHE_DURATION_HOURS', '24')),
        'validate_models': os.getenv('OPENAI_VALIDATE_MODELS', 'false').lower() == 'true',
        # GPTのみの場合の複数回答生成戦略
        'multi_response_strategy': os.getenv('OPENAI_MULTI_RESPONSE_STRATEGY', 'multi_model'),  # multi_model, multi_param, multi_perspective, gpt_variants
        'temperature_variations': [float(x) for x in os.getenv('OPENAI_TEMPERATURE_VARIATIONS', '0.3,0.7,1.0').split(',')],
        'perspectives': os.getenv('OPENAI_PERSPECTIVES', 'conservative,innovative,practical').split(','),
        # GPTバリエーション設定（モデル + reasoning.effortの組み合わせ）
        # 形式: "model1:effort1,model2:effort2" または JSON形式
        'gpt_variants': os.getenv('OPENAI_GPT_VARIANTS', 'gpt-5.2-pro:high,gpt-5.2:medium,gpt-5-mini:medium')
    },
    'gemini': {
        'enabled': os.getenv('GEMINI_ENABLED', 'false').lower() == 'true',
        'api_key': os.getenv('GEMINI_API_KEY'),
        'model': os.getenv('GEMINI_MODEL', 'gemini-pro')
    },
    'claude': {
        'enabled': os.getenv('CLAUDE_ENABLED', 'false').lower() == 'true',
        'api_key': os.getenv('CLAUDE_API_KEY'),
        'model': os.getenv('CLAUDE_MODEL', 'claude-3-opus-20240229')
    }
}

# ローカルLLM設定
LOCAL_LLM_CONFIG = {
    'provider': os.getenv('LOCAL_LLM_PROVIDER', 'ollama'),  # ollama, transformers, llama_cpp
    'models': os.getenv('LOCAL_LLM_MODELS', 'llama3:8b,mistral:7b,phi3:mini').split(','),
    'base_url': os.getenv('LOCAL_LLM_BASE_URL', 'http://localhost:11434'),  # Ollama用
    'device': os.getenv('LOCAL_LLM_DEVICE', 'cpu')  # transformers用
}

# 統合エージェント設定
INTEGRATION_CONFIG = {
    'method': os.getenv('INTEGRATION_METHOD', 'hybrid'),  # voting, weighted, meta_llm, consensus, hybrid
    'meta_llm': os.getenv('META_LLM', 'gpt-4'),  # メタLLM統合に使用するモデル
    'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', '0.7')),  # 類似度閾値
    'consensus_threshold': float(os.getenv('CONSENSUS_THRESHOLD', '0.6'))  # 合意形成閾値
}

# 調査エージェント設定
RESEARCH_CONFIG = {
    'sources': {
        'papers': {
            'enabled': os.getenv('RESEARCH_PAPERS_ENABLED', 'true').lower() == 'true',
            'sources': ['arxiv', 'pubmed', 'scholar']
        },
        'databases': {
            'enabled': os.getenv('RESEARCH_DATABASES_ENABLED', 'true').lower() == 'true',
            'sources': ['web_search', 'specialized_db']
        },
        'applications': {
            'enabled': os.getenv('RESEARCH_APPLICATIONS_ENABLED', 'true').lower() == 'true',
            'sources': ['github', 'stackoverflow', 'documentation']
        }
    },
    'max_results_per_source': int(os.getenv('RESEARCH_MAX_RESULTS_PER_SOURCE', '10'))
}

# デモモード設定
DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'

