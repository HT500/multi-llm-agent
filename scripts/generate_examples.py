"""出力例生成スクリプト - GPTのみ系統の実際の実行結果をMarkdown形式で保存"""
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents import MultiLLMAgent
from config.settings import LLM_CONFIG

def format_response(response_dict: dict) -> str:
    """レスポンスをMarkdown形式で整形"""
    label = response_dict.get('label', f"{response_dict.get('provider', 'unknown')}/{response_dict.get('model', 'unknown')}")
    model = response_dict.get('model', 'unknown')
    reasoning_effort = response_dict.get('reasoning_effort')
    temperature = response_dict.get('temperature', 0.7)
    
    header = f"### {label}\n\n"
    info = f"- **モデル**: `{model}`\n"
    if reasoning_effort:
        info += f"- **reasoning.effort**: `{reasoning_effort}`\n"
    info += f"- **temperature**: `{temperature}`\n\n"
    
    if response_dict.get('success'):
        content = response_dict.get('response', '')
        return header + info + f"{content}\n\n"
    else:
        error = response_dict.get('error', 'Unknown error')
        return header + info + f"**エラー**: {error}\n\n"

def generate_markdown(result: dict, query: str) -> str:
    """実行結果をMarkdown形式に変換"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# GPTのみ系統の出力例

## 実行情報

- **実行日時**: {timestamp}
- **質問**: {query}
- **統合方式**: {result.get('method', 'unknown')}
- **信頼度**: {result.get('confidence', 0.0):.2f}
- **生成された回答数**: {len(result.get('responses', []))}

## 使用モデルとreasoning.effortの組み合わせ

"""
    
    # 使用されたモデルとreasoning.effortの組み合わせをリストアップ
    variants_used = []
    for resp in result.get('responses', []):
        model = resp.get('model', 'unknown')
        effort = resp.get('reasoning_effort', 'none')
        label = resp.get('label', f"{model} + effort={effort}")
        variants_used.append(f"- {label}")
    
    markdown += "\n".join(variants_used) + "\n\n"
    
    # 各モデルの個別出力
    markdown += "## 各モデルの個別出力\n\n"
    for i, resp in enumerate(result.get('responses', []), 1):
        markdown += format_response(resp)
    
    # 統合後の出力
    markdown += "## 統合後の出力\n\n"
    markdown += f"{result.get('answer', '')}\n\n"
    
    # 調査結果のサマリー
    research = result.get('research_results', {})
    if research:
        markdown += "## 調査結果サマリー\n\n"
        markdown += f"- **論文数**: {len(research.get('papers', []))}\n"
        markdown += f"- **データベース結果数**: {len(research.get('databases', []))}\n"
        markdown += f"- **アプリケーション結果数**: {len(research.get('applications', []))}\n\n"
    
    return markdown

def main():
    """メイン処理"""
    # 設定確認
    if not LLM_CONFIG['openai']['enabled'] or not LLM_CONFIG['openai']['api_key']:
        print("エラー: OpenAI APIキーが設定されていません。")
        print(".envファイルにOPENAI_API_KEYを設定してください。")
        return 1
    
    # 戦略確認
    strategy = LLM_CONFIG['openai'].get('multi_response_strategy', 'multi_model')
    if strategy != 'gpt_variants':
        print(f"警告: 現在の戦略は '{strategy}' です。")
        print("GPTバリエーション戦略を使用するには、.envファイルで以下を設定してください：")
        print("OPENAI_MULTI_RESPONSE_STRATEGY=gpt_variants")
        print("OPENAI_GPT_VARIANTS=gpt-5.2-pro:high,gpt-5.2:medium,gpt-5-mini:medium")
        print("\n続行しますか？ (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            return 1
    
    # 質問の入力
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("質問を入力してください: ", end="")
        query = input().strip()
    
    if not query:
        print("エラー: 質問が入力されていません。")
        return 1
    
    print(f"\n質問: {query}")
    print("実行中...\n")
    
    # エージェントを初期化して実行
    try:
        agent = MultiLLMAgent()
        result = agent.query(query)
        
        if not result.get('success'):
            print("エラー: 回答の生成に失敗しました。")
            return 1
        
        # Markdown形式に変換
        markdown = generate_markdown(result, query)
        
        # ファイルに保存
        output_file = project_root / "examples" / "gpt_only_examples.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"\n出力例を {output_file} に保存しました。")
        print(f"\n統合後の回答:\n{result.get('answer', '')[:200]}...")
        
        return 0
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

