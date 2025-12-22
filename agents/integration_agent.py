"""統合エージェント - ハイブリッド統合方式"""
import logging
from typing import List, Dict, Optional
import numpy as np

from config.settings import INTEGRATION_CONFIG, LLM_CONFIG
from utils.similarity import SimilarityCalculator
from utils.api_clients import OpenAIClient

logger = logging.getLogger(__name__)


class IntegrationAgent:
    """統合エージェントクラス"""
    
    def __init__(self):
        self.similarity_calculator = SimilarityCalculator()
        self.meta_llm_client = None
        self._initialize_meta_llm()
    
    def _initialize_meta_llm(self):
        """メタLLMクライアントを初期化"""
        if INTEGRATION_CONFIG.get('method') in ['meta_llm', 'hybrid']:
            if LLM_CONFIG['openai']['enabled'] and LLM_CONFIG['openai']['api_key']:
                try:
                    self.meta_llm_client = OpenAIClient(LLM_CONFIG['openai']['api_key'])
                except Exception as e:
                    logger.warning(f"Failed to initialize meta LLM client: {e}")
    
    def integrate_responses(self, responses: List[Dict]) -> Dict:
        """
        複数の回答を統合
        
        Args:
            responses: 回答のリスト（各回答は {'response': str, 'provider': str, 'model': str} の形式）
            
        Returns:
            統合された回答
        """
        if not responses:
            return {'integrated_response': '', 'method': 'none', 'confidence': 0.0}
        
        if len(responses) == 1:
            return {
                'integrated_response': responses[0]['response'],
                'method': 'single',
                'confidence': 1.0,
                'source': responses[0]
            }
        
        method = INTEGRATION_CONFIG.get('method', 'hybrid')
        
        if method == 'voting':
            return self._voting_integration(responses)
        elif method == 'weighted':
            return self._weighted_integration(responses)
        elif method == 'meta_llm':
            return self._meta_llm_integration(responses)
        elif method == 'consensus':
            return self._consensus_integration(responses)
        else:  # hybrid
            return self._hybrid_integration(responses)
    
    def _voting_integration(self, responses: List[Dict]) -> Dict:
        """
        投票方式で統合（最も類似度が高い回答を選択）
        
        Args:
            responses: 回答のリスト
            
        Returns:
            統合された回答
        """
        response_texts = [r['response'] for r in responses if r.get('response')]
        
        if not response_texts:
            return {'integrated_response': '', 'method': 'voting', 'confidence': 0.0}
        
        # 類似度行列を計算
        similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(response_texts)
        
        # 各回答の平均類似度を計算（他の回答との一致度）
        avg_similarities = []
        for i in range(len(response_texts)):
            similarities = [similarity_matrix[i, j] for j in range(len(response_texts)) if j != i]
            avg_sim = np.mean(similarities) if similarities else 0.0
            avg_similarities.append(avg_sim)
        
        # 最も高い類似度を持つ回答を選択
        best_idx = np.argmax(avg_similarities)
        confidence = float(avg_similarities[best_idx])
        
        return {
            'integrated_response': response_texts[best_idx],
            'method': 'voting',
            'confidence': confidence,
            'source': responses[best_idx]
        }
    
    def _weighted_integration(self, responses: List[Dict]) -> Dict:
        """
        重み付け統合（信頼度に基づいて重み付け平均）
        
        Args:
            responses: 回答のリスト
            
        Returns:
            統合された回答
        """
        response_texts = [r['response'] for r in responses if r.get('response')]
        
        if not response_texts:
            return {'integrated_response': '', 'method': 'weighted', 'confidence': 0.0}
        
        # 各回答の信頼度を計算
        confidence_scores = []
        for i, response in enumerate(responses):
            if response.get('response'):
                other_responses = [r['response'] for j, r in enumerate(responses) if j != i and r.get('response')]
                confidence = self.similarity_calculator.calculate_confidence_score(
                    response['response'],
                    other_responses
                )
                confidence_scores.append(confidence)
            else:
                confidence_scores.append(0.0)
        
        # 信頼度を正規化（重みとして使用）
        total_confidence = sum(confidence_scores)
        if total_confidence == 0:
            weights = [1.0 / len(confidence_scores)] * len(confidence_scores)
        else:
            weights = [c / total_confidence for c in confidence_scores]
        
        # クラスタリングして代表回答を選択
        clusters = self.similarity_calculator.cluster_responses(
            response_texts,
            INTEGRATION_CONFIG.get('similarity_threshold', 0.7)
        )
        
        # 各クラスタの代表回答を選択し、重み付け
        cluster_responses = []
        cluster_weights = []
        
        for cluster_id, indices in clusters.items():
            if indices:
                rep_idx = self.similarity_calculator.find_representative_response(
                    response_texts,
                    indices
                )
                cluster_responses.append(response_texts[rep_idx])
                # クラスタの重みは、そのクラスタに属する回答の重みの合計
                cluster_weight = sum(weights[i] for i in indices)
                cluster_weights.append(cluster_weight)
        
        # 最も重みが高いクラスタの回答を選択
        if cluster_responses:
            best_cluster_idx = np.argmax(cluster_weights)
            confidence = cluster_weights[best_cluster_idx]
            
            return {
                'integrated_response': cluster_responses[best_cluster_idx],
                'method': 'weighted',
                'confidence': float(confidence),
                'cluster_info': {
                    'num_clusters': len(clusters),
                    'weights': cluster_weights
                }
            }
        else:
            return {'integrated_response': response_texts[0], 'method': 'weighted', 'confidence': 0.5}
    
    def _meta_llm_integration(self, responses: List[Dict]) -> Dict:
        """
        メタLLM統合（別のLLMが回答を統合）
        
        Args:
            responses: 回答のリスト
            
        Returns:
            統合された回答
        """
        if not self.meta_llm_client:
            logger.warning("Meta LLM client not available, falling back to weighted integration")
            return self._weighted_integration(responses)
        
        response_texts = [r['response'] for r in responses if r.get('response')]
        
        if not response_texts:
            return {'integrated_response': '', 'method': 'meta_llm', 'confidence': 0.0}
        
        # メタLLMに統合を依頼
        prompt = f"""以下の複数のLLMからの回答を統合して、最も正確で包括的な回答を作成してください。

回答1:
{response_texts[0] if len(response_texts) > 0 else ''}

回答2:
{response_texts[1] if len(response_texts) > 1 else ''}

回答3:
{response_texts[2] if len(response_texts) > 2 else ''}

これらの回答を統合し、一貫性があり、正確で、包括的な最終回答を提供してください。"""
        
        try:
            meta_model = INTEGRATION_CONFIG.get('meta_llm', 'gpt-4')
            messages = [
                {'role': 'system', 'content': 'You are an expert at integrating multiple AI responses into a coherent, accurate, and comprehensive answer.'},
                {'role': 'user', 'content': prompt}
            ]
            
            integrated_response = self.meta_llm_client.generate(meta_model, messages, temperature=0.3)
            
            return {
                'integrated_response': integrated_response,
                'method': 'meta_llm',
                'confidence': 0.8,  # メタLLM統合は高い信頼度を仮定
                'meta_model': meta_model
            }
        except Exception as e:
            logger.error(f"Meta LLM integration failed: {e}")
            return self._weighted_integration(responses)
    
    def _consensus_integration(self, responses: List[Dict]) -> Dict:
        """
        合意形成プロセスで統合
        
        Args:
            responses: 回答のリスト
            
        Returns:
            統合された回答
        """
        response_texts = [r['response'] for r in responses if r.get('response')]
        
        if not response_texts:
            return {'integrated_response': '', 'method': 'consensus', 'confidence': 0.0}
        
        # 合意スコアを計算
        consensus_score = self.similarity_calculator.calculate_consensus_score(response_texts)
        consensus_threshold = INTEGRATION_CONFIG.get('consensus_threshold', 0.6)
        
        if consensus_score >= consensus_threshold:
            # 合意が高い場合は、代表回答を選択
            clusters = self.similarity_calculator.cluster_responses(
                response_texts,
                INTEGRATION_CONFIG.get('similarity_threshold', 0.7)
            )
            
            # 最大クラスタの代表回答を選択
            largest_cluster = max(clusters.values(), key=len)
            rep_idx = self.similarity_calculator.find_representative_response(
                response_texts,
                largest_cluster
            )
            
            return {
                'integrated_response': response_texts[rep_idx],
                'method': 'consensus',
                'confidence': float(consensus_score),
                'consensus_score': float(consensus_score)
            }
        else:
            # 合意が低い場合は、メタLLMに統合を依頼
            logger.info(f"Low consensus score ({consensus_score:.2f}), using meta LLM")
            return self._meta_llm_integration(responses)
    
    def _hybrid_integration(self, responses: List[Dict]) -> Dict:
        """
        ハイブリッド統合方式
        
        1. 類似度分析でクラスタリング
        2. 各クラスタの代表回答を信頼度で重み付け
        3. 不一致が大きい場合、メタLLMが統合を実行
        4. 必要に応じて合意形成プロセスを実行
        
        Args:
            responses: 回答のリスト
            
        Returns:
            統合された回答
        """
        response_texts = [r['response'] for r in responses if r.get('response')]
        
        if not response_texts:
            return {'integrated_response': '', 'method': 'hybrid', 'confidence': 0.0}
        
        # 第一段階: 類似度分析でクラスタリング
        clusters = self.similarity_calculator.cluster_responses(
            response_texts,
            INTEGRATION_CONFIG.get('similarity_threshold', 0.7)
        )
        
        logger.info(f"Clustered {len(response_texts)} responses into {len(clusters)} clusters")
        
        # 第二段階: 各クラスタの代表回答を信頼度で重み付け
        cluster_responses = []
        cluster_confidences = []
        
        for cluster_id, indices in clusters.items():
            if indices:
                rep_idx = self.similarity_calculator.find_representative_response(
                    response_texts,
                    indices
                )
                cluster_responses.append(response_texts[rep_idx])
                
                # クラスタ内の回答の信頼度の平均
                cluster_confidence = np.mean([
                    self.similarity_calculator.calculate_confidence_score(
                        response_texts[i],
                        [response_texts[j] for j in indices if j != i]
                    )
                    for i in indices
                ])
                cluster_confidences.append(cluster_confidence)
        
        # 第三段階: 不一致が大きい場合、メタLLMが統合を実行
        if len(clusters) > 1:
            # クラスタ間の類似度が低い場合
            cluster_similarity = self.similarity_calculator.calculate_consensus_score(cluster_responses)
            
            if cluster_similarity < 0.5 and self.meta_llm_client:
                logger.info("High disagreement detected, using meta LLM for integration")
                return self._meta_llm_integration(responses)
        
        # 第四段階: 重み付け統合
        if cluster_responses:
            total_confidence = sum(cluster_confidences)
            if total_confidence > 0:
                weights = [c / total_confidence for c in cluster_confidences]
            else:
                weights = [1.0 / len(cluster_confidences)] * len(cluster_confidences)
            
            best_cluster_idx = np.argmax(weights)
            confidence = cluster_confidences[best_cluster_idx]
            
            return {
                'integrated_response': cluster_responses[best_cluster_idx],
                'method': 'hybrid',
                'confidence': float(confidence),
                'cluster_info': {
                    'num_clusters': len(clusters),
                    'cluster_similarity': float(cluster_similarity) if len(clusters) > 1 else 1.0
                }
            }
        else:
            return {'integrated_response': response_texts[0], 'method': 'hybrid', 'confidence': 0.5}

