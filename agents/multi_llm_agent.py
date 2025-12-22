"""マルチLLM統合エージェント - メインクラス"""
import logging
from typing import Dict, Optional

from agents.research_agent import ResearchAgent
from agents.llm_agents import LLMAgent
from agents.integration_agent import IntegrationAgent
from config.settings import COVERAGE_THRESHOLD, MAX_RESEARCH_ITERATIONS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiLLMAgent:
    """マルチLLM統合エージェントクラス"""
    
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.llm_agent = LLMAgent()
        self.integration_agent = IntegrationAgent()
    
    def query(self, question: str, coverage_threshold: Optional[float] = None, 
              max_iterations: Optional[int] = None) -> Dict:
        """
        質問に対して回答を生成
        
        Args:
            question: ユーザーの質問
            coverage_threshold: 網羅性閾値（オプション、デフォルトは設定値）
            max_iterations: 最大調査イテレーション数（オプション、デフォルトは設定値）
            
        Returns:
            回答情報の辞書
        """
        logger.info(f"Processing query: {question}")
        
        # 1. 調査エージェントで網羅的な調査を実行
        logger.info("Starting research phase...")
        research_results = self.research_agent.research_with_iteration(question)
        
        # 調査結果をコンテキストに変換
        context = self.research_agent.format_research_context(research_results)
        logger.info(f"Research completed. Context length: {len(context)} characters")
        
        # 2. 複数のLLMで並列に回答を生成
        logger.info("Generating responses from multiple LLMs...")
        responses = self.llm_agent.generate_responses(question, context)
        
        if not responses:
            logger.error("No responses generated")
            return {
                'answer': '申し訳ございませんが、回答を生成できませんでした。',
                'success': False,
                'responses': []
            }
        
        logger.info(f"Generated {len(responses)} responses")
        
        # 3. 統合エージェントで回答を統合
        logger.info("Integrating responses...")
        integrated = self.integration_agent.integrate_responses(responses)
        
        logger.info(f"Integration completed using method: {integrated.get('method', 'unknown')}")
        logger.info(f"Confidence: {integrated.get('confidence', 0.0):.2f}")
        
        return {
            'answer': integrated.get('integrated_response', ''),
            'success': True,
            'method': integrated.get('method', 'unknown'),
            'confidence': integrated.get('confidence', 0.0),
            'responses': responses,
            'research_results': research_results,
            'integration_info': {
                k: v for k, v in integrated.items() 
                if k not in ['integrated_response', 'method', 'confidence']
            }
        }

