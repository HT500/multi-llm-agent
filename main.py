"""メインエントリーポイント"""
import sys
import argparse
import json
from agents import MultiLLMAgent

def main():
    parser = argparse.ArgumentParser(description='マルチLLM統合エージェント')
    parser.add_argument('query', type=str, help='質問')
    parser.add_argument('--coverage-threshold', type=float, default=None,
                       help='網羅性閾値（デフォルト: 設定ファイルの値）')
    parser.add_argument('--max-iterations', type=int, default=None,
                       help='最大調査イテレーション数（デフォルト: 設定ファイルの値）')
    parser.add_argument('--output', type=str, default=None,
                       help='結果をJSONファイルに出力（オプション）')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細なログを表示')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # エージェントを初期化
    agent = MultiLLMAgent()
    
    # 質問を処理
    result = agent.query(
        args.query,
        coverage_threshold=args.coverage_threshold,
        max_iterations=args.max_iterations
    )
    
    # 結果を表示
    print("\n" + "="*80)
    print("回答")
    print("="*80)
    print(result['answer'])
    print("\n" + "="*80)
    print("詳細情報")
    print("="*80)
    print(f"統合方式: {result.get('method', 'unknown')}")
    print(f"信頼度: {result.get('confidence', 0.0):.2f}")
    print(f"生成された回答数: {len(result.get('responses', []))}")
    
    # JSON出力
    if args.output:
        # レスポンスから実際のテキストのみを抽出（サイズ削減）
        output_data = {
            'answer': result['answer'],
            'method': result.get('method'),
            'confidence': result.get('confidence'),
            'num_responses': len(result.get('responses', [])),
            'research_summary': {
                'papers': len(result.get('research_results', {}).get('papers', [])),
                'databases': len(result.get('research_results', {}).get('databases', [])),
                'applications': len(result.get('research_results', {}).get('applications', []))
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n結果を {args.output} に保存しました。")
    
    return 0 if result.get('success', False) else 1

if __name__ == '__main__':
    sys.exit(main())

