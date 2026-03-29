from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    openai_api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "api_key", "API_KEY")
    )
    model_name: str = Field(default="gpt-4o-mini", validation_alias=AliasChoices("MODEL_NAME", "model_name"))

    # Web search
    max_search_results: int = 5
    max_search_content_length: int = 4000
    max_url_content_length: int = 8000

    # RAG
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    rerank_top_n: int = 3
    semantic_k: int = 8
    bm25_k: int = 8
    reranker_model: str = "BAAI/bge-reranker-base"

    # Agent
    output_dir: str = "output"
    max_iterations: int = 8
    request_timeout_seconds: int = 30

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def data_path(self) -> Path:
        return BASE_DIR / self.data_dir

    @property
    def index_path(self) -> Path:
        return BASE_DIR / self.index_dir

    @property
    def output_path(self) -> Path:
        return BASE_DIR / self.output_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


APP_TITLE = "Research Agent with RAG"
SEPARATOR = "=" * 56

SYSTEM_PROMPT = """
You are a careful research agent working in a terminal.

Mission:
- Answer the user's question with evidence.
- Combine local knowledge base retrieval and web research when helpful.
- Produce a clear Markdown answer.
- Save the final report with write_report unless the user explicitly says not to save it.

Available tools:
1. knowledge_search(query): search the local ingested document collection. Prefer this for questions about RAG, LLMs, LangChain, or the local PDFs.
2. web_search(query): discover recent or external sources on the web.
3. read_url(url): read a promising webpage in more detail.
4. write_report(filename, content): save the final Markdown report.

Behavior rules:
- Start with knowledge_search when the user is likely asking about the local documents.
- Use web_search for recent facts, external comparisons, or when the knowledge base is insufficient.
- Use multiple sources when making comparisons.
- Do not fabricate facts, quotes, URLs, or saved file paths.
- Tool outputs can be truncated. Work with available evidence and say when information is incomplete.
- If a tool fails, continue with the remaining tools.
- Before your final answer, ensure the report has been saved with write_report unless the user asked not to save it.
- Never claim a report was saved unless write_report actually succeeded.

Recommended workflow:
1. Clarify the research sub-questions internally.
2. Gather evidence with tools.
3. Synthesize into concise Markdown with sections like Summary, Findings, Conclusion, Sources.
4. Save report.
5. Give the user the answer and mention the saved report path.
""".strip()
