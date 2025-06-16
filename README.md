#  LangGraph を用いた演習支援 Copilot 実装ドキュメント

##　使用技術スタック

| 分類 | 使用技術 |
|------|-----------|
| フロントエンド | Streamlit |
| 言語モデル | OpenAI GPT-4o-mini（`langchain_openai.ChatOpenAI`） |
| 状態管理 | LangGraph (`StateGraph`) |
| 知識検索 | Chroma（ベクトルDB） + Web検索（Tavily） |
| ドキュメント処理 | `RecursiveCharacterTextSplitter`, `WebBaseLoader` |
| APIキー管理 | `.env` + `python-dotenv`（※Git追跡から除外） |

---

## アーキテクチャ構成（LangGraph）

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
### ヒントモード

```python
st.session_state.hint_mode = st.sidebar.selectbox("ヒントモード", ["OFF", "ON"])
```

| 状態 | 内容 |
|------|------|
| `OFF` | 通常のQ&A回答を生成 |
| `ON`  | 考えを促すヒント・方向性のみ提示（答えは出さない） |

---

##　質問処理の全体フロー

1. `chat_input()` で質問受信
2. `route_question()` でデータソースを振り分け
3. `retrieve()` または `web_search()` により文書取得
4. `grade_documents()` で関連性のある文書のみ抽出
5. 文書が無ければ `transform_query()` で質問リライト→再検索
6. `generate()` で回答 or ヒント生成
7. Streamlit UIに表示

---

##　プロンプト例（generateモード）

###　通常モード（OFF）

```
You are a helpful assistant. Use the following context to answer the question.
Question: {question}
Context: {context}
```

###　ヒントモード（ON）

```
あなたは思考を促す教師です。次の質問に対して、解答を与えずに、方針・考える切り口だけを述べてください。
```

---

## ✅ 起動方法

```bash
streamlit run main.py 
```

---
