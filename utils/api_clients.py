"""LLM APIクライアント - 安全なモデル検出、キャッシュ機能、エラーハンドリング、ローカルLLM対応"""
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

from config.settings import MODEL_CACHE_FILE, LLM_CONFIG, LOCAL_LLM_CONFIG, LLM_MODE

logger = logging.getLogger(__name__)

# APIクライアント（必要に応じてインポート）
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import ollama
except ImportError:
    ollama = None


def get_available_models_cached(api_key: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    モデルリストをキャッシュ付きで取得（安全）
    
    Args:
        api_key: OpenAI APIキー
        force_refresh: キャッシュを無視して強制更新するか
        
    Returns:
        利用可能なモデル情報の辞書
    """
    # キャッシュがあれば使用
    if not force_refresh and MODEL_CACHE_FILE.exists():
        try:
            with open(MODEL_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                # キャッシュが24時間以内なら使用
                cache_age = time.time() - cached.get('timestamp', 0)
                cache_duration = LLM_CONFIG['openai'].get('cache_duration_hours', 24) * 3600
                if cache_age < cache_duration:
                    logger.info(f"Using cached model list (age: {cache_age/3600:.1f} hours)")
                    return cached['data']
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    # API呼び出しは1回だけ
    if not openai:
        logger.error("OpenAI library not installed")
        return {'models': [], 'gpt_models': [], 'latest_gpt': None, 'available_versions': []}
    
    try:
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list()
        model_ids = [model.id for model in models.data]
        
        # GPTモデルをパターンマッチングで抽出
        gpt_pattern = re.compile(r'gpt-?\d+(?:\.\d+)?(?:-turbo)?(?:-preview)?', re.IGNORECASE)
        gpt_models = [m for m in model_ids if gpt_pattern.search(m)]
        
        # バージョンでソート（最新が先頭）
        def extract_version(model_name: str) -> tuple:
            match = re.search(r'gpt-?(\d+)(?:\.(\d+))?', model_name, re.IGNORECASE)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2)) if match.group(2) else 0
                return (major, minor)
            return (0, 0)
        
        gpt_models_sorted = sorted(gpt_models, key=extract_version, reverse=True)
        
        result = {
            'models': model_ids,
            'gpt_models': gpt_models_sorted,
            'latest_gpt': gpt_models_sorted[0] if gpt_models_sorted else None,
            'available_versions': list(set([extract_version(m)[0] for m in gpt_models]))
        }
        
        # キャッシュに保存
        if LLM_CONFIG['openai'].get('cache_models', True):
            try:
                with open(MODEL_CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump({'data': result, 'timestamp': time.time()}, f, indent=2)
                logger.info("Model list cached successfully")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Failed to fetch model list: {e}")
        # エラー時はキャッシュがあれば使用
        if MODEL_CACHE_FILE.exists():
            try:
                with open(MODEL_CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)['data']
            except:
                pass
        return {'models': [], 'gpt_models': [], 'latest_gpt': None, 'available_versions': []}


def get_safe_model_list(api_key: str, preferred_models: List[str]) -> List[str]:
    """
    安全にモデルリストを取得（明示指定優先）
    
    Args:
        api_key: OpenAI APIキー
        preferred_models: 優先するモデルリスト
        
    Returns:
        利用可能なモデルのリスト
    """
    # キャッシュから利用可能モデルを確認
    available = get_available_models_cached(api_key)
    
    # 明示指定されたモデルのみを検証
    valid_models = []
    for model in preferred_models:
        model = model.strip()
        if model in available['models']:
            valid_models.append(model)
        else:
            logger.warning(f"Model {model} not found in available models, skipping")
    
    return valid_models


def validate_model(api_key: str, model_name: str) -> bool:
    """
    指定モデルが利用可能か1回だけ確認（安全）
    
    Args:
        api_key: OpenAI APIキー
        model_name: モデル名
        
    Returns:
        利用可能な場合True
    """
    if not openai:
        return False
    
    try:
        client = openai.OpenAI(api_key=api_key)
        # 軽量なテスト呼び出し（最小トークン）
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except openai.NotFoundError:
        # モデル不存在 - これは正常なエラー
        logger.debug(f"Model {model_name} not found")
        return False
    except Exception as e:
        # その他のエラーはログに記録してFalse
        logger.warning(f"Model validation error for {model_name}: {e}")
        return False


class OpenAIClient:
    """OpenAI APIクライアント"""
    
    def __init__(self, api_key: str):
        if not openai:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        self.client = openai.OpenAI(api_key=api_key)
        self.api_key = api_key
    
    def _messages_to_input(self, messages: List[Dict]) -> str:
        """
        messagesリストをinput文字列に変換（responses.create()用）
        
        Args:
            messages: メッセージリスト
            
        Returns:
            input文字列
        """
        # systemメッセージとuserメッセージを結合
        parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                parts.append(f"System: {content}")
            elif role == 'user':
                parts.append(content)
            elif role == 'assistant':
                parts.append(f"Assistant: {content}")
        
        # 最後のuserメッセージが主要なinputになる
        # または、すべてを結合
        if len(parts) == 1:
            return parts[0]
        else:
            return "\n\n".join(parts)
    
    def generate(self, model: str, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000, reasoning_effort: Optional[str] = None) -> str:
        """
        テキスト生成
        
        Args:
            model: モデル名
            messages: メッセージリスト
            temperature: 温度パラメータ
            max_tokens: 最大トークン数
            reasoning_effort: 推論努力レベル ('none', 'low', 'medium', 'high', 'xhigh')
            
        Returns:
            生成されたテキスト
        """
        # GPT-5.2系モデルかどうかを判定
        is_gpt52 = 'gpt-5' in model.lower() or 'gpt-5.2' in model.lower()
        
        # reasoning_effortが指定されている、またはGPT-5.2系の場合はresponses.create()を使用
        if reasoning_effort or is_gpt52:
            try:
                # messagesをinput文字列に変換
                input_text = self._messages_to_input(messages)
                
                # responses.create()のパラメータを構築
                params = {
                    'model': model,
                    'input': input_text,
                }
                
                # reasoning_effortが指定されている場合
                if reasoning_effort:
                    params['reasoning'] = {'effort': reasoning_effort}
                
                # temperatureはresponses.create()でも使用可能か確認
                # （サポートされていない場合は無視）
                try:
                    params['temperature'] = temperature
                except:
                    pass
                
                # max_completion_tokensを使用（responses.create()ではmax_tokensではなくmax_completion_tokens）
                params['max_completion_tokens'] = max_tokens
                
                response = self.client.responses.create(**params)
                return response.output_text
                
            except Exception as e:
                logger.warning(f"responses.create() failed for {model}, falling back to chat.completions.create(): {e}")
                # フォールバック: 通常のchat.completions.create()を使用
                # （reasoning_effortは無視）
        
        # 通常のchat.completions.create()を使用
        try:
            params = {
                'model': model,
                'messages': messages,
                'temperature': temperature,
            }
            
            # まず通常のmax_tokensで試行
            params['max_tokens'] = max_tokens
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except openai.BadRequestError as e:
            # max_tokensがサポートされていない場合（GPT-5.2など）
            error_str = str(e)
            if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                logger.info(f"Model {model} requires max_completion_tokens, retrying...")
                try:
                    # max_completion_tokensを使用して再試行
                    params = {
                        'model': model,
                        'messages': messages,
                        'temperature': temperature,
                        'max_completion_tokens': max_tokens
                    }
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message.content
                except Exception as retry_e:
                    logger.error(f"OpenAI API error (retry): {retry_e}")
                    raise
            else:
                logger.error(f"OpenAI API error: {e}")
                raise
        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise
        except openai.NotFoundError as e:
            logger.error(f"Model not found: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GeminiClient:
    """Google Gemini APIクライアント"""
    
    def __init__(self, api_key: str):
        if not genai:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self.api_key = api_key
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """
        テキスト生成
        
        Args:
            model: モデル名
            prompt: プロンプト
            temperature: 温度パラメータ
            
        Returns:
            生成されたテキスト
        """
        try:
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class ClaudeClient:
    """Anthropic Claude APIクライアント"""
    
    def __init__(self, api_key: str):
        if not anthropic:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.api_key = api_key
    
    def generate(self, model: str, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        テキスト生成
        
        Args:
            model: モデル名
            messages: メッセージリスト
            temperature: 温度パラメータ
            max_tokens: 最大トークン数
            
        Returns:
            生成されたテキスト
        """
        try:
            response = self.client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


class OllamaClient:
    """OllamaローカルLLMクライアント"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        if not ollama:
            raise ImportError("Ollama library not installed. Install with: pip install ollama")
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """
        テキスト生成
        
        Args:
            model: モデル名
            prompt: プロンプト
            temperature: 温度パラメータ
            
        Returns:
            生成されたテキスト
        """
        try:
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={'temperature': temperature}
            )
            return response['response']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def list_models(self) -> List[str]:
        """
        利用可能なモデルリストを取得
        
        Returns:
            モデル名のリスト
        """
        try:
            models = self.client.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []


def get_llm_client(provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    LLMクライアントを取得
    
    Args:
        provider: プロバイダー名（openai, gemini, claude, ollama）
        api_key: APIキー（必要に応じて）
        base_url: ベースURL（Ollama用）
        
    Returns:
        LLMクライアントインスタンス
    """
    if provider == 'openai':
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAIClient(api_key)
    elif provider == 'gemini':
        if not api_key:
            raise ValueError("Gemini API key is required")
        return GeminiClient(api_key)
    elif provider == 'claude':
        if not api_key:
            raise ValueError("Claude API key is required")
        return ClaudeClient(api_key)
    elif provider == 'ollama':
        return OllamaClient(base_url or LOCAL_LLM_CONFIG['base_url'])
    else:
        raise ValueError(f"Unknown provider: {provider}")

