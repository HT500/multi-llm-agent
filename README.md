# マルチLLM統合エージェント

複数のLLM（OpenAI、Gemini、Claude、ローカルLLM）を使用して回答を生成し、ハイブリッド方式で統合するエージェントです。網羅的な調査（論文、DB、アプリ）を網羅性ベースで繰り返し実行し、最終的に統合回答を生成します。

## 特徴

- **複数LLM対応**: OpenAI、Gemini、Claude、ローカルLLM（Ollama）に対応
- **GPTのみでも動作**: 同一APIで異なるモデル/パラメータで複数回答を生成
- **安全なモデル検出**: キャッシュ機能、明示指定優先、全当たり探索なし
- **網羅的な調査**: 論文、データベース、アプリケーションを網羅的に調査
- **ハイブリッド統合**: 複数の統合方式（投票、重み付け、メタLLM、合意形成）を組み合わせ

## インストール

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd new_project
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env`ファイルを作成し、以下の環境変数を設定してください：

```bash
# OpenAI API（必須）
OPENAI_API_KEY=your_openai_api_key_here

# オプション: Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_ENABLED=false

# オプション: Claude API
CLAUDE_API_KEY=your_claude_api_key_here
CLAUDE_ENABLED=false

# LLMモード: "api" または "local"
LLM_MODE=api

# ローカルLLM設定（LLM_MODE=localの場合）
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODELS=llama3:8b,mistral:7b,phi3:mini

# 網羅性設定
COVERAGE_THRESHOLD=0.8
MAX_RESEARCH_ITERATIONS=5

# 統合方式: voting, weighted, meta_llm, consensus, hybrid
INTEGRATION_METHOD=hybrid

# メタLLM設定（統合に使用するLLM）
META_LLM_PROVIDER=openai  # openai, claude, gemini
META_LLM_MODEL=gpt-5.2  # または gpt-5.2-pro, claude-3-opus-20240229, gemini-3-pro-preview
```

## 使用方法

### 基本的な使用

```bash
python main.py "量子コンピューティングの最新の進展について教えてください"
```

### オプション付きの使用

```bash
# 網羅性閾値と最大イテレーション数を指定
python main.py "質問" --coverage-threshold 0.9 --max-iterations 3

# 結果をJSONファイルに出力
python main.py "質問" --output result.json

# 詳細なログを表示
python main.py "質問" --verbose
```

### 出力例の生成（GPTバリエーション）

GPTバリエーション戦略で実行し、各モデルの個別出力と統合結果をMarkdown形式で保存：

```bash
# .envでOPENAI_MULTI_RESPONSE_STRATEGY=gpt_variantsを設定後
python scripts/run_query.py "質問内容"
```

または、対話的に質問を入力：

```bash
python scripts/run_query.py
```

結果は`responses/response_質問の一部_タイムスタンプ.md`に保存されます。

### Pythonコードから使用

```python
from agents import MultiLLMAgent

# エージェントを初期化
agent = MultiLLMAgent()

# 質問を処理
result = agent.query(
    "量子コンピューティングの最新の進展について教えてください",
    coverage_threshold=0.8,
    max_iterations=5
)

# 結果を表示
print(result['answer'])
print(f"統合方式: {result['method']}")  # メタLLM使用時は "meta_llm (GPT: gpt-5.2)" のように表示
print(f"信頼度: {result['confidence']}")
```

## 設定

### GPTのみで使用する場合

`.env`ファイルで以下のように設定：

```bash
OPENAI_API_KEY=your_api_key
OPENAI_ENABLED=true
OPENAI_MODELS=gpt-5.2,gpt-4,gpt-3.5-turbo
OPENAI_AUTO_DISCOVER=false
OPENAI_MULTI_RESPONSE_STRATEGY=multi_model  # multi_model, multi_param, multi_perspective, gpt_variants

GEMINI_ENABLED=false
CLAUDE_ENABLED=false
```

### GPTバリエーション戦略（モデル + reasoning.effort）

GPT-5.2 Proやreasoning.effortパラメータを使用する場合：

```bash
OPENAI_MULTI_RESPONSE_STRATEGY=gpt_variants
OPENAI_GPT_VARIANTS=gpt-5.2-pro:high,gpt-5.2:medium,gpt-5-mini:medium
```

**形式**: `モデル名:reasoning.effort`をカンマ区切りで指定

**reasoning.effortの値**:
- `none`: 推論なし
- `low`: 低
- `medium`: 標準（デフォルト）
- `high`: 高
- `xhigh`: 最高

**例**:
- `gpt-5.2-pro:high` → GPT-5.2 Pro + Thinking高
- `gpt-5.2:medium` → GPT-5.2 + Thinking標準
- `gpt-5-mini:medium` → GPT-5 Mini + Thinking標準

各バリエーションの個別出力と統合結果が生成されます。

### ローカルLLMを使用する場合

1. Ollamaをインストール（https://ollama.ai/）

2. モデルをダウンロード：

```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull phi3:mini
```

3. `.env`ファイルで設定：

```bash
LLM_MODE=local
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODELS=llama3:8b,mistral:7b,phi3:mini
```

## アーキテクチャ

### 主要コンポーネント

1. **調査エージェント（Research Agent）**
   - 論文検索（arXiv、PubMed、Google Scholar）
   - データベース検索（Web検索、専門DB）
   - アプリケーション調査（GitHub、Stack Overflow、ドキュメント）
   - 網羅性評価と繰り返し制御

2. **LLM回答生成エージェント（LLM Agent）**
   - 複数のLLMで並列回答生成
   - GPTのみの場合、異なるモデル/パラメータで複数回答生成
   - コンテキスト注入

3. **統合エージェント（Integration Agent）**
   - ハイブリッド統合方式
   - 類似度分析、信頼度スコアリング
   - 重み付け統合、メタLLM統合、合意形成
   - メタLLMは`.env`でプロバイダー（OpenAI/Claude/Gemini）とモデルを設定可能

### システムフロー

```
ユーザー質問
    ↓
調査エージェント（網羅性チェック）
    ↓
利用可能モデル確認
    ↓
複数LLM並列実行 / GPT複数設定実行 / ローカルLLM実行
    ↓
統合エージェント
    ↓
最終回答
```

## セキュリティとベストプラクティス

### Banリスク対策

- ✅ モデルリスト取得は起動時1回のみ（キャッシュ使用）
- ✅ 存在しないモデルは試行しない
- ✅ レート制限エラー時の適切な待機処理
- ✅ 明示的なモデル指定を優先

### コスト最適化

- ✅ キャッシュで不要なAPI呼び出しを削減
- ✅ モデル検証はオプション（デフォルト無効）
- ✅ エラー時の即座なスキップ

## トラブルシューティング

### OpenAI APIエラー

- APIキーが正しく設定されているか確認
- モデル名が正しいか確認（`gpt-5.2`など）
- レート制限に達していないか確認

### ローカルLLMが動作しない

- Ollamaが起動しているか確認: `ollama list`
- モデルがダウンロードされているか確認: `ollama list`
- ベースURLが正しいか確認（デフォルト: `http://localhost:11434`）

### 調査結果が少ない

- 網羅性閾値を下げる（`COVERAGE_THRESHOLD=0.6`など）
- 最大イテレーション数を増やす（`MAX_RESEARCH_ITERATIONS=10`など）

## ライセンス

[ライセンス情報を記載]

## 貢献

[貢献方法を記載]

