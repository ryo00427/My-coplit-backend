import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List, Annotated, Literal, Sequence, TypedDict
from langgraph.graph import END, StateGraph, START
import asyncio
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

class RouteQuery(BaseModel):
    """ユーザーのクエリを最も関連性の高いデータソースにルーティングします。"""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="ユーザーの質問に応じて、ウェブ検索またはベクターストアにルーティングします。",
    )


class GradeDocuments(BaseModel):
    """取得された文書の関連性チェックのためのバイナリスコア。"""

    binary_score: str = Field(
        description="文書が質問に関連しているかどうか、「yes」または「no」"
    )


class GradeHallucinations(BaseModel):
    """生成された回答における幻覚の有無を示すバイナリスコア。"""

    binary_score: str = Field(
        description="回答が事実に基づいているかどうか、「yes」または「no」"
    )

class GradeAnswer(BaseModel):
    """回答が質問に対処しているかどうかを評価するバイナリスコア。"""

    binary_score: str = Field(
        description="回答が質問に対処しているかどうか、「yes」または「no」"
    )

