from typing import Literal
from pydantic import BaseModel, Field


# =============================================================================
# RAGAS plugin config (Mode 1: Eval-only).
#
# Mode 1: user supplies a pre-generated RAG dataset (4-column: user_input,
# retrieved_contexts, response, reference) + judge LLM credentials. Plugin
# calls ragas.evaluate() directly. NO RAG built. NO PDF processed.
#
# Mode 2 (build-then-eval from PDFs) is COMMENTED OUT below. Re-enable when
# customer scenario requires plugin to build a baseline RAG from raw docs.
# See RAGAS_AISC_Reference.docx section 6 for Mode 1 vs Mode 2 tradeoffs.
# =============================================================================


class RagasConfig(BaseModel):
    # -------------------------------------------------------------------------
    # Judge LLM — the LLM ragas uses to score each (Q, ctx, answer, truth) row
    # -------------------------------------------------------------------------
    evaluator_llm_provider: Literal[
        "openai",
        "anthropic",
        "google",
        "mistral",
        "openai_compat",
    ] = Field(
        default="openai",
        description=(
            "Which provider runs the judge LLM that scores each row. "
            "openai, anthropic, google, and mistral call those vendors' official cloud APIs. "
            "openai_compat points at any OpenAI-compatible endpoint, so it covers local servers "
            "like Ollama, vLLM, or llama.cpp (set the base URL for it)."
        ),
    )
    evaluator_llm_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "Name of the judge model, passed straight through to the provider. "
            "Examples by provider: 'gpt-4o-mini' (openai, fast and cheap, and reliably completes all 5 metrics), "
            "'claude-3-5-sonnet-latest' (anthropic), 'gemini-1.5-flash' (google), "
            "'mistral-large-latest' (mistral), and 'qwen2.5:7b' or 'llama3:latest' (openai_compat / Ollama). "
            "Heads up on local models: each question needs several sequential judge calls, and the heavier "
            "metrics (faithfulness, noise_sensitivity, context_precision) can time out on small 7B-8B models "
            "and come back as NaN. For full, dependable coverage, reach for a hosted model like OpenAI or Anthropic."
        ),
    )
    evaluator_llm_api_key: str = Field(
        default="",
        description=(
            "API key for the judge LLM provider. Leave it blank to fall back to the matching key from .env "
            "or the environment. openai_compat endpoints usually take the base URL instead of a key."
        ),
    )
    evaluator_llm_base_url: str = Field(
        default="",
        description=(
            "Endpoint URL for the openai_compat provider, for example http://localhost:11434/v1 for Ollama. "
            "It is only read when the provider is openai_compat and is ignored otherwise."
        ),
    )

    # -------------------------------------------------------------------------
    # Embeddings — used by ragas metrics like answer_relevancy that compute
    # cosine similarity between texts. Independent of judge LLM.
    # -------------------------------------------------------------------------
    embeddings_provider: Literal["hf_local", "openai"] = Field(
        default="hf_local",
        description=(
            "Where the embeddings come from. Some metrics, like response relevancy, compare texts by "
            "cosine similarity and need an embedding model, separate from the judge LLM. "
            "hf_local runs a HuggingFace SentenceTransformer on the machine itself, which is free but slower. "
            "openai calls the OpenAI embeddings API, which is faster but paid and needs a key."
        ),
    )
    embeddings_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description=(
            "Name of the embedding model, passed straight to the chosen provider. "
            "For hf_local, try 'BAAI/bge-small-en-v1.5' or 'sentence-transformers/all-MiniLM-L6-v2'. "
            "For openai, try 'text-embedding-3-small' or 'text-embedding-3-large'."
        ),
    )
    embeddings_api_key: str = Field(
        default="",
        description=(
            "API key for the embeddings provider. Needed when the provider is openai, where you can also "
            "leave it blank to fall back to OPENAI_API_KEY from .env or the environment. "
            "Leave it empty for hf_local."
        ),
    )

    prompt_language: str = Field(
        default="english",
        description=(
            "Label for the language of the dataset. This is just used to group and tag results in the "
            "dashboard, it does not change how anything is scored, so any free-text label works."
        ),
    )

    max_workers: int = Field(
        default=1,
        description=(
            "How many evaluation samples ragas works on in parallel. Keep it at 1 for rate-limited APIs or "
            "local models, and raise it (say 5 to 10) to speed up runs against cloud providers that can "
            "handle the concurrency."
        ),
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
