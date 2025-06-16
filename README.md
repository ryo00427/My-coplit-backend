# \U0001F680 LangGraph を用いた演習支援 Copilot 実装ドキュメント

## \U0001F3AF プロジェクト概要

本プロジェクトは、大学のAI講義における「実装演習の自立支援」を目的として開発された **対話型AI Copilot** です。受講生の質問に対して、文書検索（RAG）・ヒント提示・評価ループを通じて「答えを与える」のではなく「学習を促す」設計を採用しています。

---

## \U0001F6E0️ 使用技術スタック

| 分類 | 使用技術 |
|------|-----------|
| フロントエンド | Streamlit |
| 言語モデル | OpenAI GPT-4o-mini（`langchain_openai.ChatOpenAI`） |
| 状態管理 | LangGraph (`StateGraph`) |
| 知識検索 | Chroma（ベクトルDB） + Web検索（Tavily） |
| ドキュメント処理 | `RecursiveCharacterTextSplitter`, `WebBaseLoader` |
| APIキー管理 | `.env` + `python-dotenv`（※Git追跡から除外） |

---

## \U0001F9E9 アーキテクチャ構成（LangGraph）

各ステップをノードとして構成し、ユーザーの質問に対して条件付きでフローを制御します。

### ノード構成

- `route_question`: vectorstore or web_search への振り分け
- `retrieve`: ドキュメントのRAG検索
- `web_search`: TavilyによるWeb検索
- `grade_documents`: ドキュメントの関連性を判定
- `transform_query`: 質問のリライト
- `generate`: 回答生成 or ヒント提示
- `END`: 回答表示・終了

### 状態型

```python
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
```

---

## \U0001F4A1 ヒントモード

```python
st.session_state.hint_mode = st.sidebar.selectbox("ヒントモード", ["OFF", "ON"])
```

| 状態 | 内容 |
|------|------|
| `OFF` | 通常のQ&A回答を生成 |
| `ON`  | 考えを促すヒント・方向性のみ提示（答えは出さない） |

---

## \U0001F501 質問処理の全体フロー

1. `chat_input()` で質問受信
2. `route_question()` でデータソースを振り分け
3. `retrieve()` または `web_search()` により文書取得
4. `grade_documents()` で関連性のある文書のみ抽出
5. 文書が無ければ `transform_query()` で質問リライト→再検索
6. `generate()` で回答 or ヒント生成
7. Streamlit UIに表示

---

## \U0001F9EA プロンプト例（generateモード）

### 通常モード（OFF）

```
You are a helpful assistant. Use the following context to answer the question.
Question: {question}
Context: {context}
```

### ヒントモード（ON）

```
あなたは思考を促す教師です。次の質問に対して、解答を与えずに、方針・考える切り口だけを述べてください。
```

---

## \U0001F512 セキュリティとキー管理

- `.env` に `OPENAI_API_KEY` を定義
- `.gitignore` に `.env` を必ず追加
- GitHub Push Protection により `.env` の push は禁止
- `.env.example` にダミー値を記載し、チームで共有

---

## ✅ 起動方法

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## \U0001F9F1 今後の拡張アイデア

- ヒント＋回答の同時表示（タブ形式）
- 学習ログ保存（Firestore連携など）
- 質問カテゴリ自動分類（分類ノードの追加）
- Slack連携Bot化 or FastAPIでAPI提供化

---

## \U0001F4DD 注意

このアプリは教育演習支援用であり、医療・法律・セキュリティ領域では使用しないでください。
