"""調査エージェント - 論文/DB/アプリ検索、繰り返し制御"""
import logging
from typing import List, Dict, Optional
import arxiv
import requests
from config.settings import (
    COVERAGE_THRESHOLD, MAX_RESEARCH_ITERATIONS, RESEARCH_CONFIG
)
from utils.coverage_evaluator import CoverageEvaluator

logger = logging.getLogger(__name__)


class ResearchAgent:
    """調査エージェントクラス"""
    
    def __init__(self):
        self.coverage_evaluator = CoverageEvaluator()
        self.research_history = []  # 調査履歴
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        論文を検索
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            
        Returns:
            論文情報のリスト
        """
        results = []
        
        # arXiv検索
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                results.append({
                    'title': paper.title,
                    'abstract': paper.summary,
                    'authors': [author.name for author in paper.authors],
                    'published': paper.published.isoformat(),
                    'url': paper.entry_id,
                    'source': 'arxiv'
                })
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
        
        # PubMed検索（簡易実装）
        # 実際の実装では、PubMed APIを使用
        try:
            # ここでは簡易的な実装
            pass
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
        
        # Google Scholar検索（簡易実装）
        # 実際の実装では、scholarlyライブラリを使用
        try:
            # ここでは簡易的な実装
            pass
        except Exception as e:
            logger.warning(f"Google Scholar search failed: {e}")
        
        return results
    
    def search_databases(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        データベースを検索
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            
        Returns:
            データベース検索結果のリスト
        """
        results = []
        
        # Web検索（簡易実装）
        # 実際の実装では、Google Custom Search APIやSerpAPIを使用
        try:
            # ここでは簡易的な実装
            # 実際にはAPIを使用してWeb検索を実行
            pass
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
        
        # 専門DB検索（簡易実装）
        # 実際の実装では、各専門DBのAPIを使用
        try:
            # ここでは簡易的な実装
            pass
        except Exception as e:
            logger.warning(f"Specialized DB search failed: {e}")
        
        return results
    
    def search_applications(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        アプリケーションを検索
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            
        Returns:
            アプリケーション検索結果のリスト
        """
        results = []
        
        # GitHub検索（簡易実装）
        # 実際の実装では、GitHub APIを使用
        try:
            # ここでは簡易的な実装
            # 実際にはGitHub APIを使用してリポジトリを検索
            pass
        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")
        
        # Stack Overflow検索（簡易実装）
        # 実際の実装では、Stack Exchange APIを使用
        try:
            # ここでは簡易的な実装
            pass
        except Exception as e:
            logger.warning(f"Stack Overflow search failed: {e}")
        
        # ドキュメント検索（簡易実装）
        try:
            # ここでは簡易的な実装
            pass
        except Exception as e:
            logger.warning(f"Documentation search failed: {e}")
        
        return results
    
    def conduct_research(self, query: str, iteration: int = 0) -> Dict[str, List[Dict]]:
        """
        調査を実行
        
        Args:
            query: 検索クエリ
            iteration: イテレーション番号
            
        Returns:
            調査結果の辞書
        """
        results = {
            'papers': [],
            'databases': [],
            'applications': []
        }
        
        max_results = RESEARCH_CONFIG.get('max_results_per_source', 10)
        
        # 論文検索
        if RESEARCH_CONFIG['sources']['papers']['enabled']:
            try:
                papers = self.search_papers(query, max_results)
                results['papers'] = papers
                logger.info(f"Found {len(papers)} papers")
            except Exception as e:
                logger.error(f"Paper search error: {e}")
        
        # データベース検索
        if RESEARCH_CONFIG['sources']['databases']['enabled']:
            try:
                db_results = self.search_databases(query, max_results)
                results['databases'] = db_results
                logger.info(f"Found {len(db_results)} database results")
            except Exception as e:
                logger.error(f"Database search error: {e}")
        
        # アプリケーション検索
        if RESEARCH_CONFIG['sources']['applications']['enabled']:
            try:
                app_results = self.search_applications(query, max_results)
                results['applications'] = app_results
                logger.info(f"Found {len(app_results)} application results")
            except Exception as e:
                logger.error(f"Application search error: {e}")
        
        return results
    
    def research_with_iteration(self, query: str) -> Dict[str, List[Dict]]:
        """
        網羅性を満たすまで調査を繰り返し実行
        
        Args:
            query: 検索クエリ
            
        Returns:
            統合された調査結果
        """
        all_results = {
            'papers': [],
            'databases': [],
            'applications': []
        }
        
        previous_results = None
        
        for iteration in range(MAX_RESEARCH_ITERATIONS):
            logger.info(f"Research iteration {iteration + 1}/{MAX_RESEARCH_ITERATIONS}")
            
            # 調査を実行
            current_results = self.conduct_research(query, iteration)
            
            # 結果を統合
            for source_type in all_results:
                all_results[source_type].extend(current_results[source_type])
            
            # 網羅性を評価
            evaluation = self.coverage_evaluator.evaluate_coverage(
                all_results,
                previous_results
            )
            
            logger.info(f"Coverage score: {evaluation['overall_score']:.2f}")
            logger.info(f"  - Diversity: {evaluation['diversity_score']:.2f}")
            logger.info(f"  - Balance: {evaluation['balance_score']:.2f}")
            logger.info(f"  - New info rate: {evaluation['new_info_rate']:.2f}")
            
            # 網羅性が閾値を超えたら終了
            if evaluation['overall_score'] >= COVERAGE_THRESHOLD:
                logger.info(f"Coverage threshold ({COVERAGE_THRESHOLD}) reached. Stopping research.")
                break
            
            # 未探索領域を特定して次の検索に活用
            gaps = self.coverage_evaluator.detect_gaps(all_results, query)
            if gaps:
                logger.info(f"Detected gaps: {gaps}")
                # 次のイテレーションでギャップを埋めるためのクエリを調整
                # （簡易実装では、同じクエリを使用）
            
            previous_results = all_results.copy()
        
        return all_results
    
    def format_research_context(self, research_results: Dict[str, List[Dict]]) -> str:
        """
        調査結果をコンテキスト形式に整形
        
        Args:
            research_results: 調査結果の辞書
            
        Returns:
            整形されたコンテキスト文字列
        """
        context_parts = []
        
        # 論文情報
        if research_results['papers']:
            context_parts.append("## 論文情報\n")
            for i, paper in enumerate(research_results['papers'][:5], 1):  # 上位5件
                context_parts.append(f"{i}. {paper.get('title', 'N/A')}")
                context_parts.append(f"   {paper.get('abstract', '')[:200]}...")
                context_parts.append(f"   出典: {paper.get('source', 'N/A')}\n")
        
        # データベース情報
        if research_results['databases']:
            context_parts.append("## データベース情報\n")
            for i, db_result in enumerate(research_results['databases'][:5], 1):
                context_parts.append(f"{i}. {db_result.get('title', 'N/A')}")
                context_parts.append(f"   {db_result.get('content', '')[:200]}...\n")
        
        # アプリケーション情報
        if research_results['applications']:
            context_parts.append("## アプリケーション情報\n")
            for i, app_result in enumerate(research_results['applications'][:5], 1):
                context_parts.append(f"{i}. {app_result.get('title', 'N/A')}")
                context_parts.append(f"   {app_result.get('description', '')[:200]}...\n")
        
        return "\n".join(context_parts)

