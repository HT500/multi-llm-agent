"""LLM回答生成エージェント - 並列実行、コンテキスト注入、GPTのみでも複数回答生成"""
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from config.settings import LLM_CONFIG, LOCAL_LLM_CONFIG, LLM_MODE
from utils.api_clients import (
    get_llm_client, get_safe_model_list, OpenAIClient
)

logger = logging.getLogger(__name__)


class LLMAgent:
    """LLM回答生成エージェントクラス"""
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """LLMクライアントを初期化"""
        # OpenAI
        if LLM_CONFIG['openai']['enabled'] and LLM_CONFIG['openai']['api_key']:
            try:
                self.clients['openai'] = OpenAIClient(LLM_CONFIG['openai']['api_key'])
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Gemini
        if LLM_CONFIG['gemini']['enabled'] and LLM_CONFIG['gemini']['api_key']:
            try:
                self.clients['gemini'] = get_llm_client('gemini', LLM_CONFIG['gemini']['api_key'])
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        
        # Claude
        if LLM_CONFIG['claude']['enabled'] and LLM_CONFIG['claude']['api_key']:
            try:
                self.clients['claude'] = get_llm_client('claude', LLM_CONFIG['claude']['api_key'])
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
        
        # ローカルLLM
        if LLM_MODE == 'local':
            try:
                self.clients['local'] = get_llm_client(
                    LOCAL_LLM_CONFIG['provider'],
                    base_url=LOCAL_LLM_CONFIG['base_url']
                )
            except Exception as e:
                logger.error(f"Failed to initialize local LLM client: {e}")
    
    def _generate_single_response(self, provider: str, model: str, messages: List[Dict], 
                                  temperature: float = 0.7, perspective: Optional[str] = None) -> Dict:
        """
        単一の回答を生成
        
        Args:
            provider: プロバイダー名
            model: モデル名
            messages: メッセージリスト
            temperature: 温度パラメータ
            perspective: 視点（保守的、革新的、実用的など）
            
        Returns:
            回答情報の辞書
        """
        try:
            # 視点に応じてsystem promptを調整
            if perspective:
                system_prompt = self._get_perspective_prompt(perspective)
                if messages and messages[0].get('role') == 'system':
                    messages[0]['content'] = system_prompt
                else:
                    messages.insert(0, {'role': 'system', 'content': system_prompt})
            
            client = self.clients[provider]
            
            if provider == 'openai':
                response_text = client.generate(model, messages, temperature)
            elif provider == 'gemini':
                # Gemini用にメッセージをプロンプトに変換
                prompt = self._messages_to_prompt(messages)
                response_text = client.generate(model, prompt, temperature)
            elif provider == 'claude':
                response_text = client.generate(model, messages, temperature)
            elif provider == 'local':
                prompt = self._messages_to_prompt(messages)
                response_text = client.generate(model, prompt, temperature)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            return {
                'provider': provider,
                'model': model,
                'response': response_text,
                'temperature': temperature,
                'perspective': perspective,
                'success': True
            }
        except Exception as e:
            logger.error(f"Failed to generate response from {provider}/{model}: {e}")
            return {
                'provider': provider,
                'model': model,
                'response': None,
                'error': str(e),
                'success': False
            }
    
    def _get_perspective_prompt(self, perspective: str) -> str:
        """
        視点に応じたsystem promptを取得
        
        Args:
            perspective: 視点（conservative, innovative, practical）
            
        Returns:
            system prompt
        """
        prompts = {
            'conservative': "You are a conservative and cautious analyst. Provide well-established, evidence-based answers with careful consideration of risks and limitations.",
            'innovative': "You are an innovative and forward-thinking analyst. Provide creative and cutting-edge insights, exploring new possibilities and emerging trends.",
            'practical': "You are a practical and pragmatic analyst. Provide actionable, real-world solutions with focus on implementation and feasibility."
        }
        return prompts.get(perspective, prompts['practical'])
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        メッセージリストをプロンプト文字列に変換
        
        Args:
            messages: メッセージリスト
            
        Returns:
            プロンプト文字列
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)
    
    def _get_multi_response_strategy(self) -> List[Dict]:
        """
        GPTのみの場合の複数回答生成戦略を取得
        
        Returns:
            回答生成設定のリスト
        """
        strategy = LLM_CONFIG['openai'].get('multi_response_strategy', 'multi_model')
        strategies = []
        
        if strategy == 'multi_model':
            # 異なるモデルを使用
            models = LLM_CONFIG['openai']['models']
            for model in models[:3]:  # 最大3つ
                strategies.append({
                    'provider': 'openai',
                    'model': model,
                    'temperature': 0.7,
                    'perspective': None
                })
        
        elif strategy == 'multi_param':
            # 同じモデルで異なるパラメータを使用
            model = LLM_CONFIG['openai']['models'][0] if LLM_CONFIG['openai']['models'] else 'gpt-4'
            temperatures = LLM_CONFIG['openai'].get('temperature_variations', [0.3, 0.7, 1.0])
            for temp in temperatures:
                strategies.append({
                    'provider': 'openai',
                    'model': model,
                    'temperature': temp,
                    'perspective': None
                })
        
        elif strategy == 'multi_perspective':
            # 異なる視点のsystem promptを使用
            model = LLM_CONFIG['openai']['models'][0] if LLM_CONFIG['openai']['models'] else 'gpt-4'
            perspectives = LLM_CONFIG['openai'].get('perspectives', ['conservative', 'innovative', 'practical'])
            for perspective in perspectives:
                strategies.append({
                    'provider': 'openai',
                    'model': model,
                    'temperature': 0.7,
                    'perspective': perspective
                })
        
        return strategies
    
    def generate_responses(self, query: str, context: str = "") -> List[Dict]:
        """
        複数のLLMで並列に回答を生成
        
        Args:
            query: ユーザーの質問
            context: 調査エージェントが収集した情報（コンテキスト）
            
        Returns:
            回答のリスト
        """
        # 日本語で回答するかどうかを判定
        import re
        is_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', query))
        language_instruction = "Please respond in Japanese (日本語で回答してください)." if is_japanese else "Please respond in the same language as the question."
        
        # メッセージを構築
        messages = []
        if context:
            messages.append({
                'role': 'system',
                'content': f"You are a helpful assistant. {language_instruction} Use the following research context to answer the user's question accurately and comprehensively.\n\n{context}"
            })
        else:
            messages.append({
                'role': 'system',
                'content': f"You are a helpful assistant. {language_instruction}"
            })
        messages.append({
            'role': 'user',
            'content': query
        })
        
        # 使用可能なプロバイダーを確認
        available_providers = list(self.clients.keys())
        
        # GPTのみの場合の戦略を決定
        if len(available_providers) == 1 and 'openai' in available_providers:
            logger.info("Only OpenAI available, using multi-response strategy")
            strategies = self._get_multi_response_strategy()
        else:
            # 複数プロバイダーがある場合
            strategies = []
            
            # OpenAI
            if 'openai' in available_providers:
                models = get_safe_model_list(
                    LLM_CONFIG['openai']['api_key'],
                    LLM_CONFIG['openai']['models']
                )
                for model in models[:1]:  # 最初の利用可能なモデルのみ
                    strategies.append({
                        'provider': 'openai',
                        'model': model,
                        'temperature': 0.7,
                        'perspective': None
                    })
            
            # Gemini
            if 'gemini' in available_providers:
                strategies.append({
                    'provider': 'gemini',
                    'model': LLM_CONFIG['gemini']['model'],
                    'temperature': 0.7,
                    'perspective': None
                })
            
            # Claude
            if 'claude' in available_providers:
                strategies.append({
                    'provider': 'claude',
                    'model': LLM_CONFIG['claude']['model'],
                    'temperature': 0.7,
                    'perspective': None
                })
            
            # ローカルLLM
            if 'local' in available_providers:
                for model in LOCAL_LLM_CONFIG['models']:
                    strategies.append({
                        'provider': 'local',
                        'model': model,
                        'temperature': 0.7,
                        'perspective': None
                    })
        
        # 並列実行
        responses = []
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = {}
            for strategy in strategies:
                future = executor.submit(
                    self._generate_single_response,
                    strategy['provider'],
                    strategy['model'],
                    messages.copy(),
                    strategy['temperature'],
                    strategy.get('perspective')
                )
                futures[future] = strategy
            
            for future in as_completed(futures):
                try:
                    response = future.result()
                    if response['success']:
                        responses.append(response)
                except Exception as e:
                    logger.error(f"Response generation failed: {e}")
        
        logger.info(f"Generated {len(responses)} responses from {len(strategies)} strategies")
        return responses

