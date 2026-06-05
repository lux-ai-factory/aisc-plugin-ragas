from typing import Literal
from pydantic import BaseModel, Field


# =============================================================================
# MLA-RAGAS plugin config (Mode 1: Eval-only).
#
# Mode 1: user supplies a pre-generated RAG dataset (4-column: user_input,
# retrieved_contexts, response, reference) + judge LLM credentials. Plugin
# calls ragas.evaluate() directly. NO RAG built. NO PDF processed.
#
# Mode 2 (build-then-eval from PDFs) is COMMENTED OUT below. Re-enable when
# customer scenario requires plugin to build a baseline RAG from raw docs.
# See RAGAS_AISC_Reference.docx section 6 for Mode 1 vs Mode 2 tradeoffs.
# =============================================================================


class MLARagasConfig(BaseModel):
    # -------------------------------------------------------------------------
    # Judge LLM — the LLM ragas uses to score each (Q, ctx, answer, truth) row
    # -------------------------------------------------------------------------
    evaluator_llm_provider: Literal[
        "openai",
        "anthropic",
        "google",
        "mistral",
        "openai_compat",
        "llm_factory",
    ] = Field(
        default="openai",
        description=(
            "Provider for the judge LLM. "
            "openai/anthropic/google/mistral use the official cloud API. "
            "openai_compat = any OpenAI-API-compatible endpoint (Ollama, vLLM, llama.cpp). "
            "llm_factory = internal proxy."
        ),
    )
    evaluator_llm_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "Model name. Examples: 'gpt-4o-mini' (openai, fast/cheap, all 5 metrics complete), "
            "'claude-3-5-sonnet-latest' (anthropic), 'gemini-1.5-flash' (google), "
            "'mistral-large-latest' (mistral), 'qwen2.5:7b' (openai_compat / Ollama — faster than llama3 on Mac), "
            "'llama3:latest' (openai_compat / Ollama — slow on heavy metrics like faithfulness, may return NaN), "
            "'OpenAIGPT4o' (llm_factory). "
            "Tradeoff: smaller local models (8B) often timeout on faithfulness + noise_sensitivity + context_precision "
            "(each needs 3–5 sequential LLM calls per question). Use OpenAI/Anthropic for full coverage."
        ),
    )
    evaluator_llm_api_key: str = Field(
        default="",
        description="API key for the judge LLM provider. Not needed for llm_factory.",
    )
    evaluator_llm_base_url: str = Field(
        default="",
        description=(
            "Base URL for openai_compat provider (e.g. http://localhost:11434/v1 for Ollama). "
            "Ignored for other providers."
        ),
    )
    llm_factory_url: str = Field(
        default="http://host.docker.internal:5001",
        description="Base URL of the LLM Factory service. Only used when evaluator_llm_provider=llm_factory.",
    )

    # -------------------------------------------------------------------------
    # Embeddings — used by ragas metrics like answer_relevancy that compute
    # cosine similarity between texts. Independent of judge LLM.
    # -------------------------------------------------------------------------
    embeddings_provider: Literal["hf_local", "openai"] = Field(
        default="hf_local",
        description=(
            "Embedding provider. hf_local = HuggingFace SentenceTransformer running locally (free, slower). "
            "openai = OpenAI embeddings API (faster, paid)."
        ),
    )
    embeddings_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description=(
            "Embedding model name. HF examples: 'BAAI/bge-small-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2'. "
            "OpenAI examples: 'text-embedding-3-small', 'text-embedding-3-large'."
        ),
    )
    embeddings_api_key: str = Field(
        default="",
        description="API key for embeddings provider. Required when embeddings_provider=openai. Empty for hf_local.",
    )

    prompt_language: str = Field(
        default="english",
        description="The language of the prompts in the dataset. Used for grouping and labeling results in the dashboard.",
    )

    max_workers: int = Field(
        default=1,
        description="The maximum number of parallel workers for Ragas evaluation. Set to 1 for rate-limited APIs or local models, or increase (e.g., 5-10) for faster cloud executions.",
        ge=1,
    )


# =============================================================================
# === MODE 2 (commented out — build RAG from PDFs then evaluate) ==============
# =============================================================================
# Re-enable the following fields when plugin needs to construct a baseline RAG
# from a corpus of PDFs (chunk -> embed -> FAISS -> retrieve -> generate -> eval).
# Inputs needed in Mode 2: pdf-files + dataset (Q + ground_truth only).
#
# Pipeline LLM (the LLM IN the constructed RAG that generates answers):
#
# pipeline_llm_source: Literal["llm_factory", "direct"] = Field(
#     default="llm_factory",
#     description='LLM source for RAG answer generation: "llm_factory" routes through LLM Factory; "direct" uses provider API key directly',
# )
# pipeline_llm_model: str = Field(
#     default="OpenAIGPT4o",
#     description=(
#         'LLM Factory model key (e.g. "OpenAIGPT4o") when pipeline_llm_source="llm_factory", '
#         'or provider model name (e.g. "mistral-medium") when pipeline_llm_source="direct"'
#     ),
# )
# pipeline_llm_type: str = Field(
#     default="mistral",
#     description=(
#         'Used only when pipeline_llm_source="direct". '
#         'One of: "mistral", "openai", "hf-text-generation", "hf-text2text-generation"'
#     ),
# )
#
# Translate/paraphrase LLM (used for multilingual experiments):
#
# translate_llm_model: str = Field(
#     default="OpenAIGPT4o",
#     description="LLM Factory model key used for translation and paraphrase generation",
# )
#
# RAG-pipeline config (chunker + retriever + generator parameters):
#
# embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="HuggingFace embedding model name")
# chunk_size: int = Field(default=400, description="Document chunk size in characters")
# chunk_overlap: int = Field(default=40, description="Chunk overlap in characters")
# top_k: int = Field(default=3, description="Number of documents to retrieve per query")
# temperature: float = Field(default=0.2, description="LLM sampling temperature")
# max_tokens: int = Field(default=512, description="Maximum tokens for LLM responses")
#
# Multilingual settings:
#
# source_language: str = Field(default="english", description="Source language of the knowledge base and dataset")
# target_languages: list[str] = Field(default=[], description='Target languages for multilingual evaluation. Empty = single language only.')
# num_paraphrases: int = Field(default=0, description="Number of paraphrases per question to generate and evaluate (0 = no paraphrasing)")
# strategy: str = Field(
#     default="crosslingual_answer",
#     description='Multilingual strategy: "crosslingual_answer" = translate questions only, answer in target language; "translate_docs" = translate document chunks before indexing',
# )
# translate_context: bool = Field(default=True, description="Whether to translate retrieved context snippets before passing to the LLM")
#
# Direct-mode API keys (separate from Mode-1 evaluator_llm_api_key):
#
# openai_api_key: str = Field(default="", description="OpenAI API key (only needed for direct OpenAI mode)")
# mistral_api_key: str = Field(default="", description="Mistral API key (only needed for direct Mistral mode)")
# =============================================================================
