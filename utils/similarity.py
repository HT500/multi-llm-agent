"""類似度計算ユーティリティ"""
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """類似度計算クラス"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        テキスト間の類似度行列を計算
        
        Args:
            texts: テキストのリスト
            
        Returns:
            類似度行列（n×n）
        """
        if len(texts) < 2:
            return np.array([[1.0]])
        
        try:
            # TF-IDFベクトル化
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # コサイン類似度を計算
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
        except Exception as e:
            logger.warning(f"Failed to calculate similarity matrix: {e}")
            # エラー時は単位行列を返す
            return np.eye(len(texts))
    
    def calculate_pairwise_similarities(self, texts: List[str]) -> List[float]:
        """
        ペアワイズ類似度を計算
        
        Args:
            texts: テキストのリスト
            
        Returns:
            ペアワイズ類似度のリスト
        """
        similarity_matrix = self.calculate_similarity_matrix(texts)
        
        # 上三角行列からペアワイズ類似度を抽出
        similarities = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i, j])
        
        return similarities
    
    def cluster_responses(self, responses: List[str], similarity_threshold: float = 0.7) -> Dict[int, List[int]]:
        """
        回答をクラスタリング（類似度ベース）
        
        Args:
            responses: 回答のリスト
            similarity_threshold: 類似度閾値
            
        Returns:
            クラスタID -> インデックスのリストの辞書
        """
        if len(responses) < 2:
            return {0: [0]} if responses else {}
        
        try:
            similarity_matrix = self.calculate_similarity_matrix(responses)
            
            # 距離行列に変換（1 - 類似度）
            distance_matrix = 1 - similarity_matrix
            
            # 負の値を0にクリップ（数値誤差対策）
            distance_matrix = np.clip(distance_matrix, 0, None)
            
            # DBSCANでクラスタリング
            # eps: 距離閾値、min_samples: 最小サンプル数
            clustering = DBSCAN(eps=1 - similarity_threshold, min_samples=1, metric='precomputed')
            labels = clustering.fit_predict(distance_matrix)
            
            # クラスタごとにインデックスをグループ化
            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
            
            return clusters
        except Exception as e:
            logger.warning(f"Failed to cluster responses: {e}")
            # エラー時は各回答を個別のクラスタに
            return {i: [i] for i in range(len(responses))}
    
    def calculate_consensus_score(self, responses: List[str]) -> float:
        """
        回答間の合意スコアを計算
        
        Args:
            responses: 回答のリスト
            
        Returns:
            合意スコア（0-1、高いほど合意度が高い）
        """
        if len(responses) < 2:
            return 1.0
        
        try:
            similarities = self.calculate_pairwise_similarities(responses)
            
            # 平均類似度を合意スコアとする
            consensus_score = np.mean(similarities)
            
            return float(consensus_score)
        except Exception as e:
            logger.warning(f"Failed to calculate consensus score: {e}")
            return 0.5
    
    def find_representative_response(self, responses: List[str], cluster_indices: List[int]) -> int:
        """
        クラスタの代表回答を選択（他の回答との平均類似度が最も高いもの）
        
        Args:
            responses: 回答のリスト
            cluster_indices: クラスタに属するインデックスのリスト
            
        Returns:
            代表回答のインデックス
        """
        if len(cluster_indices) == 1:
            return cluster_indices[0]
        
        try:
            similarity_matrix = self.calculate_similarity_matrix(responses)
            
            # 各回答の平均類似度を計算
            avg_similarities = []
            for idx in cluster_indices:
                cluster_similarities = [similarity_matrix[idx, j] for j in cluster_indices if j != idx]
                avg_sim = np.mean(cluster_similarities) if cluster_similarities else 0.0
                avg_similarities.append(avg_sim)
            
            # 平均類似度が最も高いものを代表とする
            best_idx = np.argmax(avg_similarities)
            
            return cluster_indices[best_idx]
        except Exception as e:
            logger.warning(f"Failed to find representative response: {e}")
            return cluster_indices[0]
    
    def calculate_confidence_score(self, response: str, other_responses: List[str]) -> float:
        """
        回答の信頼度スコアを計算（他の回答との一致度）
        
        Args:
            response: 対象の回答
            other_responses: 他の回答のリスト
            
        Returns:
            信頼度スコア（0-1）
        """
        if not other_responses:
            return 0.5  # 比較対象がない場合はデフォルト値
        
        try:
            all_responses = [response] + other_responses
            similarity_matrix = self.calculate_similarity_matrix(all_responses)
            
            # 最初の回答（対象回答）と他の回答との類似度の平均
            similarities = similarity_matrix[0, 1:]
            confidence_score = np.mean(similarities)
            
            return float(confidence_score)
        except Exception as e:
            logger.warning(f"Failed to calculate confidence score: {e}")
            return 0.5

