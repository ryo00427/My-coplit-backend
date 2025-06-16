import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List, Literal, TypedDict
from langgraph.graph import END, START
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(...)

class GradeDocuments(BaseModel):
    binary_score: str = Field(...)

class GradeHallucinations(BaseModel):
    binary_score: str = Field(...)

class GradeAnswer(BaseModel):
    binary_score: str = Field(...)

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

async def route_question(state):
    st.session_state.status.update(label="**---ROUTE QUESTION---**", state="running", expanded=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", "ベクターストアにはagent, prompt engineering, adversarial attackの文書が含まれています。該当すればvectorstore、なければweb_searchにルーティングしてください。"),
        ("human", "{question}"),
    ])
    question_router = route_prompt | structured_llm_router
    question = state["question"]
    source = question_router.invoke({"question": question})
    return source.datasource

async def retrieve(state):
    embd = OpenAIEmbeddings()
    urls = [
        "https://zenn.dev/meu/articles/27dd04851c01c4",
        "https://zenn.dev/tsuzukia/articles/0724729c2b733e",
        "https://zenn.dev/umi_mori/books/prompt-engineer",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)
    vectorstore = Chroma.from_documents(doc_splits, collection_name="rag-chroma", embedding=embd)
    retriever = vectorstore.as_retriever()
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}

async def web_search(state):
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": state["question"]})
    web_results = "\n".join([d["content"] for d in docs])
    return {"documents": [Document(page_content=web_results)], "question": state["question"]}

async def grade_documents(state):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "質問に関連する文書か評価してください。yes または no で。"),
        ("human", "Document: {document}\nQuestion: {question}"),
    ])
    grader = grade_prompt | structured_llm_grader
    question = state["question"]
    filtered_docs = []
    for d in state["documents"]:
        score = grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)
    return {"documents": filtered_docs, "question": question}

async def transform_query(state):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", "質問をベクトル検索向けに改善してください。"),
        ("human", "{question}"),
    ])
    rewriter = re_write_prompt | llm | StrOutputParser()
    better_question = rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": better_question}

async def decide_to_generate(state):
    return "generate" if state["documents"] else "transform_query"

async def generate(state):
    st.session_state.status.update(label="**---GENERATE---**", state="running", expanded=False)
    if st.session_state.hint_mode == "ON":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "解答ではなくヒントや考える方向性を提示してください。"),
            ("human", "Question: {question}\nContext: {context}"),
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "コンテキストをもとに質問に答えてください。"),
            ("human", "Question: {question}\nContext: {context}"),
        ])
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation}

async def run_workflow(inputs):
    st.session_state.status = st.status(label="**実行中...**", expanded=True)
    st.session_state.placeholder = st.empty()
    value = await st.session_state.workflow.ainvoke(inputs)
    st.session_state.status.update(label="**完了！**", state="complete")
    st.session_state.placeholder.markdown(value["generation"])

def st_rag_langgraph():
    st.title("演習支援 Copilot by LangGraph")

    # 🔧 UI: ヒントモード選択
    st.session_state.hint_mode = st.sidebar.selectbox("ヒントモード", ["OFF", "ON"])

    if not hasattr(st.session_state, "workflow"):
        workflow = StateGraph(GraphState)
        workflow.add_node("web_search", web_search)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("transform_query", transform_query)
        workflow.add_conditional_edges(START, route_question, {"vectorstore": "retrieve", "web_search": "web_search"})
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", decide_to_generate, {"generate": "generate", "transform_query": "transform_query"})
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)
        st.session_state.workflow = workflow.compile()

    if prompt := st.chat_input("質問を入力してください"):
        with st.chat_message("user"): st.markdown(prompt)
        inputs = {"question": prompt}
        asyncio.run(run_workflow(inputs))

if __name__ == "__main__":
    st_rag_langgraph()
