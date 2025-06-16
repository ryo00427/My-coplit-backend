# 演習支援 Copilot - LangGraph による対話型AIシステム

## 概要

このアプリケーションは、LangChain と LangGraph を活用して実装された、AIベースの演習支援 Copilot です。学生が質問を自然言語で入力することで、文書検索（RAG）および生成AIを用いた回答またはヒントを提供します。

---

## 使用技術

- **Streamlit**：チャットUIとヒント切り替えのインターフェース
- **LangChain + LangGraph**：状態遷移による制御構成
- **OpenAI GPT-4o-mini**：応答生成および各種判定
- **Chroma DB**：ベクトル検索による情報検索（RAG）
- **Tavily API**：外部Web検索結果取得（fallback）

---

## 構成概要（LangGraph）

```mermaid
graph TD
    Start --> Route
    Route -->|vectorstore| Retrieve
    Route -->|web_search| WebSearch
    Retrieve --> GradeDocs
    GradeDocs -->|documentsあり| Generate
    GradeDocs -->|documentsなし| TransformQuery
    TransformQuery --> Retrieve
    WebSearch --> Generate
    Generate --> End
