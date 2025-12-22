"""網羅性評価モジュール - 多様性スコア、カバレッジマップ"""
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class CoverageEvaluator:
    """網羅性評価クラス"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    def calculate_diversity_score(self, texts: List[str]) -> float:
        """
        検索結果の多様性スコアを計算（TF-IDFベース）
        
        Args:
            texts: テキストのリスト
            
        Returns:
            多様性スコア（0-1、高いほど多様）
        """
        if len(texts) < 2:
            return 0.0
        
        try:
            # TF-IDFベクトル化
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # コサイン類似度を計算
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 対角成分（自己類似度）を除いた平均類似度
            # 類似度が低いほど多様性が高い
            mask = ~np.eye(len(texts), dtype=bool)
            avg_similarity = similarity_matrix[mask].mean()
            
            # 多様性スコア = 1 - 平均類似度
            diversity_score = 1.0 - avg_similarity
            
            return max(0.0, min(1.0, diversity_score))
        except Exception as e:
            logger.warning(f"Failed to calculate diversity score: {e}")
            return 0.5  # デフォルト値
    
    def calculate_coverage_map(self, research_results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        情報源のカバレッジマップを計算
        
        Args:
            research_results: 調査結果の辞書
                {
                    'papers': [...],
                    'databases': [...],
                    'applications': [...]
                }
        
        Returns:
            各情報源のカバレッジスコア
        """
        coverage_map = {}
        total_results = 0
        
        # 各情報源の結果数をカウント
        for source_type, results in research_results.items():
            count = len(results) if results else 0
            coverage_map[source_type] = count
            total_results += count
        
        # 正規化（各情報源の割合）
        if total_results > 0:
            for source_type in coverage_map:
                coverage_map[source_type] = coverage_map[source_type] / total_results
        else:
            # 結果がない場合は均等に
            for source_type in coverage_map:
                coverage_map[source_type] = 1.0 / len(coverage_map) if coverage_map else 0.0
        
        return coverage_map
    
    def calculate_new_information_rate(self, previous_results: List[str], current_results: List[str]) -> float:
        """
        新規情報の発見率を計算
        
        Args:
            previous_results: 前回の調査結果
            current_results: 現在の調査結果
            
        Returns:
            新規情報の割合（0-1）
        """
        if not current_results:
            return 0.0
        
        previous_set = set(previous_results)
        current_set = set(current_results)
        
        # 新規情報 = 現在の結果に含まれるが前回の結果に含まれないもの
        new_items = current_set - previous_set
        
        new_rate = len(new_items) / len(current_set) if current_set else 0.0
        
        return new_rate
    
    def detect_gaps(self, research_results: Dict[str, List[Dict]], query: str) -> List[Tuple[str, float]]:
        """
        未探索領域を特定
        
        Args:
            research_results: 調査結果の辞書
            query: 検索クエリ
            
        Returns:
            (情報源タイプ, 優先度) のリスト
        """
        coverage_map = self.calculate_coverage_map(research_results)
        
        # カバレッジが低い情報源を優先
        gaps = []
        for source_type, coverage in coverage_map.items():
            priority = 1.0 - coverage  # カバレッジが低いほど優先度が高い
            gaps.append((source_type, priority))
        
        # 優先度でソート
        gaps.sort(key=lambda x: x[1], reverse=True)
        
        return gaps
    
    def evaluate_coverage(self, research_results: Dict[str, List[Dict]], 
                         previous_results: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, float]:
        """
        網羅性を総合評価
        
        Args:
            research_results: 現在の調査結果
            previous_results: 前回の調査結果（オプション）
            
        Returns:
            網羅性評価結果
        """
        # 全テキストを収集
        all_texts = []
        for results in research_results.values():
            for result in results:
                if isinstance(result, dict):
                    text = result.get('title', '') + ' ' + result.get('abstract', '') + ' ' + result.get('content', '')
                    all_texts.append(text)
        
        # 多様性スコア
        diversity_score = self.calculate_diversity_score(all_texts) if all_texts else 0.0
        
        # カバレッジマップ
        coverage_map = self.calculate_coverage_map(research_results)
        
        # カバレッジバランススコア（各情報源が均等にカバーされているか）
        coverage_values = list(coverage_map.values())
        if coverage_values:
            # 標準偏差が小さいほどバランスが良い
            std_dev = np.std(coverage_values)
            balance_score = 1.0 / (1.0 + std_dev)  # 0-1に正規化
        else:
            balance_score = 0.0
        
        # 新規情報率
        new_info_rate = 0.0
        if previous_results:
            prev_texts = []
            for results in previous_results.values():
                for result in results:
                    if isinstance(result, dict):
                        text = result.get('title', '') + ' ' + result.get('abstract', '') + ' ' + result.get('content', '')
                        prev_texts.append(text)
            
            new_info_rate = self.calculate_new_information_rate(prev_texts, all_texts)
        
        # 総合スコア（重み付け平均）
        overall_score = (
            diversity_score * 0.4 +
            balance_score * 0.3 +
            new_info_rate * 0.3
        )
        
        return {
            'overall_score': overall_score,
            'diversity_score': diversity_score,
            'balance_score': balance_score,
            'new_info_rate': new_info_rate,
            'coverage_map': coverage_map
        }

