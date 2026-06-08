"""RAGAS plugin — Mode 1 (eval-only).

User supplies a 4-column dataset (user_input, retrieved_contexts, response,
reference) + judge LLM credentials + embedding choice. Plugin calls
ragas.evaluate() directly. NO RAG built. NO PDF processed.

Mode 2 (build-RAG-from-PDFs-then-evaluate) is preserved as a comment block
at the bottom of this file. See RAGAS_AISC_Reference.docx section 6 for the
tradeoffs and the rationale for keeping Mode 2 reachable.
"""
from __future__ import annotations

import io as _io
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

# .env is deliberately NOT loaded into os.environ globally to prevent cross-plugin
# credential leakage within the Celery worker process.

# Monkeypatch langchain_mistralai to fix TypeError: unsupported operand type(s) for +=: 'dict' and 'dict'
try:
    def _safe_combine_dict(d1, d2):
        res = d1.copy()
        for k, v in d2.items():
            if k in res:
                if isinstance(res[k], dict) and isinstance(v, dict):
                    res[k] = _safe_combine_dict(res[k], v)
                elif isinstance(res[k], (int, float)) and isinstance(v, (int, float)):
                    res[k] += v
            else:
                res[k] = v
        return res

    def _patched_combine_llm_outputs(self, llm_outputs: list) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                continue
            token_usage = output.get("token_usage")
            if token_usage is not None:
                overall_token_usage = _safe_combine_dict(overall_token_usage, token_usage)
        return {"token_usage": overall_token_usage, "model_name": self.model}

    from langchain_mistralai.chat_models import ChatMistralAI
    ChatMistralAI._combine_llm_outputs = _patched_combine_llm_outputs
except Exception:
    pass

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    ChartType,
    InputType,
    MetricVisualization,
    TaskProgress,
    evaluation_input,
    metric,
)
from aisc_plugin_interface.models.measure import Measure

from .form_schema import RagasConfig
from .json_input_provider import RawBytesProvider


def _error_result(msg: str) -> dict:
    print(f"[RAGAS] ERROR: {msg}", file=sys.stderr)
    return {"error": msg, "success": False, "summary": [], "full_results": []}


def _load_dataset_frame(raw: bytes) -> pd.DataFrame:
    """Turn the uploaded dataset bytes into a DataFrame, accepting JSON or CSV.

    We sniff the content rather than trust a file extension: a JSON dataset
    parses cleanly into a DataFrame, and anything that isn't JSON falls back to
    CSV. Either way we end up with the same 4 columns (user_input,
    retrieved_contexts, response, reference). JSON keeps retrieved_contexts as a
    native list per row; CSV keeps the existing JSON-array / '||'-delimited
    conventions, which parse_contexts handles downstream.

    Accepted JSON shapes:
      - a top-level list of row objects: [{"user_input": ...}, ...]
      - an object wrapping the rows under "data"/"rows"/"samples"/"records"
      - an object of equal-length columns: {"user_input": [...], ...}
    """
    import json

    text = raw.decode("utf-8", errors="replace").lstrip("﻿").strip()

    # Try JSON first. Only the obvious JSON openers are worth attempting, so a
    # normal CSV (which never starts with { or [) skips straight to read_csv.
    if text[:1] in ("[", "{"):
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError(
                f"Dataset looked like JSON but failed to parse: {exc}"
            ) from exc

        rows = parsed
        if isinstance(parsed, dict):
            for key in ("data", "rows", "samples", "records"):
                if isinstance(parsed.get(key), list):
                    rows = parsed[key]
                    break

        if isinstance(rows, list):
            if not rows:
                raise ValueError("JSON dataset is empty (no rows).")
            return pd.DataFrame(rows)
        if isinstance(rows, dict):
            # Column-oriented object, e.g. {"user_input": [...], "response": [...]}.
            return pd.DataFrame(rows)
        raise ValueError(
            "Unrecognized JSON dataset shape. Expected a list of row objects, "
            "an object with a 'data'/'rows'/'samples'/'records' list, or an "
            "object of equal-length columns."
        )

    # Not JSON -> treat as CSV (the original, still-supported path).
    return pd.read_csv(_io.BytesIO(raw))


@evaluation_input(
    name="dataset-file",
    label="RAG eval dataset, CSV or JSON (columns: user_input, retrieved_contexts, response, reference)",
    input_provider_class=RawBytesProvider,
    input_type=InputType.DATASET,
    required=True,
)
class RagasPlugin(BaseEvaluationPlugin[RagasConfig]):
    plugin_name = "RAGAS"
    ui_icon = "assessment"

    form_ui_schema = {
        "evaluator_llm_api_key": {"ui:widget": "password"},
        "embeddings_api_key": {"ui:widget": "password"},
        "evaluator_llm_base_url": {"ui:placeholder": "http://localhost:11434/v1"},
    }

    def on_config_change(self, form_data):
        """Show only the credential fields that the current providers actually use.

        The judge-LLM and embeddings sections each carry fields that only make
        sense for one provider. Rather than show everything and hope the user
        ignores the irrelevant boxes, we drop the unused ones from the schema and
        from form_data so nothing stale gets persisted.

        Rules:
          - evaluator_llm_provider == "openai_compat": show the base URL, hide the
            API key (a local/compat endpoint usually needs the URL, not a key).
            Any other provider: hide the base URL, show the API key.
          - embeddings_api_key is only relevant for embeddings_provider == "openai".
        """
        schema, ui_schema = self.get_full_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])

        def read(field: str, default: str) -> str:
            if isinstance(form_data, dict):
                return form_data.get(field) or default
            if form_data is not None:
                return getattr(form_data, field, default) or default
            return default

        def hide(field: str) -> None:
            props.pop(field, None)
            if field in required:
                required.remove(field)
            ui_schema.setdefault(field, {})["ui:widget"] = "hidden"
            if isinstance(form_data, dict):
                form_data.pop(field, None)

        evaluator_provider = read("evaluator_llm_provider", "openai")
        embeddings_provider = read("embeddings_provider", "hf_local")

        if evaluator_provider == "openai_compat":
            hide("evaluator_llm_api_key")
        else:
            hide("evaluator_llm_base_url")

        if embeddings_provider != "openai":
            hide("embeddings_api_key")

        return form_data, schema, ui_schema

    def evaluate(self, config_data: dict) -> Any:
        import sys
        try:
            return self._run_evaluation(config_data)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[RAGAS] Unhandled exception:\n{tb}", file=sys.stderr)
            return _error_result(str(exc))

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        # No charts. The 5 aggregate means already render as KPI cards at the top
        # of the results, and the full per-sample breakdown — every column
        # (user_input, response, retrieved_contexts, reference, and each metric's
        # score) — is the `ragas_evaluation.csv` artifact, which renders as a
        # complete table. A measurement-based per-sample table can only show
        # name/score and drops those columns, so we don't render it here.
        return [
            MetricVisualization(chart_type=ChartType.TABLE, metrics=[
                "RAGAS Overall Score",
                "RAGAS Run Success",
            ]),
        ]

    def _run_evaluation(self, config_data: dict) -> Any:
        config = self.validate_config_form_data(config_data)

        self.report_progress(TaskProgress(progress=0.05, extra={"stage": "setup"}))

        dataset_bytes: bytes | None = self.get_input_data("dataset-file")
        if not dataset_bytes:
            return _error_result("No dataset uploaded.")

        try:
            dataset_df = _load_dataset_frame(dataset_bytes)
        except Exception as exc:
            return _error_result(f"Failed to parse dataset: {exc}")

        required_cols = {"user_input", "retrieved_contexts", "response", "reference"}
        missing = required_cols - set(dataset_df.columns)
        if missing:
            return _error_result(
                f"Dataset is missing required columns: {sorted(missing)}. "
                "Mode 1 expects columns: user_input, retrieved_contexts, response, reference."
            )

        self.report_progress(TaskProgress(progress=0.15, extra={"stage": "dataset_loaded"}))
        print(f"[RAGAS] Dataset loaded: {len(dataset_df)} samples.", file=sys.stderr)

        self.report_progress(TaskProgress(progress=0.25, extra={"stage": "building_judge_llm"}))
        judge_llm = self._build_judge_llm(config)
        embeddings = self._build_embeddings(config)

        self.report_progress(TaskProgress(progress=0.40, extra={"stage": "ragas_evaluating"}))
        result_df = self._run_ragas(dataset_df, judge_llm, embeddings, config)

        self.report_progress(TaskProgress(progress=0.90, extra={"stage": "aggregating"}))
        non_metric_cols = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts", "language"}
        
        if "language" in dataset_df.columns:
            result_df["language"] = dataset_df["language"].values
        else:
            result_df["language"] = config.prompt_language
            
        full_with_lang = result_df.copy()
        cols = ["language"] + [c for c in full_with_lang.columns if c != "language"]
        full_with_lang = full_with_lang[cols]

        metric_cols = [
            c for c in result_df.columns
            if c not in non_metric_cols and pd.api.types.is_numeric_dtype(result_df[c])
        ]
        
        summary_rows = []
        all_nan = True
        
        for lang, group in result_df.groupby("language"):
            summary_row = {"language": lang}
            for c in metric_cols:
                mean_val = group[c].mean()
                if pd.notna(mean_val):
                    summary_row[c] = float(mean_val)
                    all_nan = False
                else:
                    summary_row[c] = None
            summary_rows.append(summary_row)
            
        summary_df = pd.DataFrame(summary_rows)

        if metric_cols and all_nan:
            return _error_result(
                "All ragas metrics returned NaN — every judge LLM call failed. "
                "Check evaluator_llm_provider / model / base_url / api_key (trim whitespace) "
                "and verify the endpoint is reachable from inside the eval-worker container."
            )

        self.report_progress(TaskProgress(progress=0.95, extra={"stage": "uploading_artifacts"}))
        self._upload_dataframe(full_with_lang, "ragas_evaluation.csv")
        self._upload_dataframe(summary_df, "ragas_summary.csv")

        return {
            "success": True,
            "full_results": full_with_lang.to_dict(orient="records"),
            "summary": summary_df.to_dict(orient="records"),
            "languages_evaluated": summary_df["language"].tolist(),
        }

    # -----------------------------------------------------------------------
    # Provider dispatch
    # -----------------------------------------------------------------------

    def _read_secure_env(self, key_name: str) -> str:
        """Safely extracts a key from .env without polluting global memory."""
        from pathlib import Path
        dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if dotenv_path.exists():
            with open(dotenv_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == key_name:
                            return v.strip().strip('"').strip("'")
        return ""

    def _build_judge_llm(self, config: RagasConfig):
        provider = (config.evaluator_llm_provider or "").strip()
        model = (config.evaluator_llm_model or "").strip()
        api_key = (config.evaluator_llm_api_key or "").strip()

        # Fall back to secure local env or global environment if form field is empty
        if not api_key:
            import os
            if provider == "openai":
                api_key = self._read_secure_env("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
            elif provider == "anthropic":
                api_key = self._read_secure_env("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
            elif provider == "google":
                api_key = self._read_secure_env("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            elif provider == "mistral":
                api_key = self._read_secure_env("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY", "")
            api_key = api_key.strip()

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, api_key=api_key or None, temperature=0, timeout=60, max_retries=10)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, api_key=api_key or None, temperature=0, default_request_timeout=60, max_retries=10)
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key or None, temperature=0, timeout=60, max_retries=10)
        if provider == "mistral":
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(model=model, api_key=api_key or None, temperature=0, timeout=60, max_retries=10)
        if provider == "openai_compat":
            from langchain_openai import ChatOpenAI
            base_url = (config.evaluator_llm_base_url or "http://localhost:11434/v1").strip()
            return ChatOpenAI(
                model=model,
                api_key=api_key or "no-key",
                base_url=base_url,
                temperature=0,
                timeout=60,
                max_retries=10,
            )
        raise ValueError(f"Unknown evaluator_llm_provider: {provider}")

    def _build_embeddings(self, config: RagasConfig):
        provider = (config.embeddings_provider or "").strip()
        model = (config.embeddings_model or "").strip()
        api_key = (config.embeddings_api_key or "").strip()

        if not api_key and provider == "openai":
            import os
            api_key = (self._read_secure_env("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()

        if provider == "hf_local":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model)
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model, api_key=api_key or None)
        raise ValueError(f"Unknown embeddings_provider: {provider}")

    # -----------------------------------------------------------------------
    # Ragas wrapping
    # -----------------------------------------------------------------------

    def _run_ragas(self, dataset_df: pd.DataFrame, judge_llm, embeddings, config: RagasConfig) -> pd.DataFrame:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            NoiseSensitivity,
            ResponseRelevancy,
        )

        def parse_contexts(v):
            if isinstance(v, list):
                return [str(x) for x in v]
            if pd.isna(v):
                return []
            s = str(v)
            try:
                import json
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            return [c.strip() for c in s.split("||") if c.strip()]

        data = {
            "user_input": dataset_df["user_input"].astype(str).tolist(),
            "response": dataset_df["response"].astype(str).tolist(),
            "retrieved_contexts": dataset_df["retrieved_contexts"].map(parse_contexts).tolist(),
            "reference": dataset_df["reference"].astype(str).tolist(),
        }
        if "reference_contexts" in dataset_df.columns:
            data["reference_contexts"] = dataset_df["reference_contexts"].map(parse_contexts).tolist()

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[
                ContextPrecision(),
                ContextRecall(),
                NoiseSensitivity(),
                ResponseRelevancy(),
                Faithfulness(),
            ],
            llm=LangchainLLMWrapper(judge_llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
            run_config=RunConfig(max_workers=config.max_workers, max_retries=10, max_wait=60)
        )
        return result.to_pandas()

    def _upload_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        self.upload_artifact(filename, csv_bytes)

    # -----------------------------------------------------------------------
    # Metric reporters (per-language preserved for output compat)
    # -----------------------------------------------------------------------

    @metric("RAGAS Run Success")
    def run_success(self, evaluation_output: Any) -> list[Measure]:
        success = evaluation_output.get("success", False)
        err = evaluation_output.get("error") if not success else None
        return [
            Measure(
                name="RAGAS Run Success",
                score=1.0 if success else 0.0,
                description="Indicates if the evaluation run was successful (1.0) or unsuccessful (0.0).",
                error=err,
            )
        ]

    @metric("RAGAS Overall Score")
    def overall_score(self, evaluation_output: Any) -> list[Measure]:
        summary = evaluation_output.get("summary", [])
        if not summary:
            return [Measure(name="RAGAS Overall Score", score=0.0)]
        aliases = {
            "faithfulness": ["faithfulness"],
            "context_recall": ["context_recall"],
            "context_precision": ["context_precision"],
            "noise_sensitivity": ["noise_sensitivity"],
            "response_relevancy": ["response_relevancy", "answer_relevancy"],
        }
        measures = []
        for row in summary:
            scores = []
            for key, alist in aliases.items():
                for col in row:
                    for a in alist:
                        if col == a or col.startswith(a + "("):
                            v = row[col]
                            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                                scores.append(float(v))
                            break
                    else:
                        continue
                    break
            avg = sum(scores) / len(scores) if scores else 0.0
            measures.append(Measure(
                name="RAGAS Overall Score",
                score=round(avg, 4),
                description="The overall RAG performance score (ranging from 0.0 to 1.0), calculated as the mean of all evaluated RAGAS metrics. A higher score indicates better alignment across retrieval accuracy and generation quality.",
            ))
        return measures

    @metric("RAGAS Faithfulness")
    def faithfulness(self, evaluation_output: Any) -> list[Measure]:
        return self._mean_metric(evaluation_output, "faithfulness", "RAGAS Faithfulness")

    @metric("RAGAS Context Recall")
    def context_recall(self, evaluation_output: Any) -> list[Measure]:
        return self._mean_metric(evaluation_output, "context_recall", "RAGAS Context Recall")

    @metric("RAGAS Context Precision")
    def context_precision(self, evaluation_output: Any) -> list[Measure]:
        return self._mean_metric(evaluation_output, "context_precision", "RAGAS Context Precision")

    @metric("RAGAS Noise Sensitivity")
    def noise_sensitivity(self, evaluation_output: Any) -> list[Measure]:
        return self._mean_metric(evaluation_output, "noise_sensitivity", "RAGAS Noise Sensitivity")

    @metric("RAGAS Response Relevancy")
    def response_relevancy(self, evaluation_output: Any) -> list[Measure]:
        return self._mean_metric(evaluation_output, "response_relevancy", "RAGAS Response Relevancy")

    # ---- per-sample metrics (one Measurement per question per metric) ----
    # Each registered metric is a separate name so the dashboard's
    # metric_visualizations TABLE chart can pull JUST the per-sample rows
    # while BARS/RADAR pull JUST the aggregate rows.

    @metric("RAGAS Faithfulness Per Sample")
    def per_sample_faithfulness(self, evaluation_output: Any) -> list[Measure]:
        return self._per_sample(evaluation_output, "faithfulness", "RAGAS Faithfulness Per Sample")

    @metric("RAGAS Context Recall Per Sample")
    def per_sample_context_recall(self, evaluation_output: Any) -> list[Measure]:
        return self._per_sample(evaluation_output, "context_recall", "RAGAS Context Recall Per Sample")

    @metric("RAGAS Context Precision Per Sample")
    def per_sample_context_precision(self, evaluation_output: Any) -> list[Measure]:
        return self._per_sample(evaluation_output, "context_precision", "RAGAS Context Precision Per Sample")

    @metric("RAGAS Noise Sensitivity Per Sample")
    def per_sample_noise_sensitivity(self, evaluation_output: Any) -> list[Measure]:
        return self._per_sample(evaluation_output, "noise_sensitivity", "RAGAS Noise Sensitivity Per Sample")

    @metric("RAGAS Response Relevancy Per Sample")
    def per_sample_response_relevancy(self, evaluation_output: Any) -> list[Measure]:
        return self._per_sample(evaluation_output, "response_relevancy", "RAGAS Response Relevancy Per Sample")

    def _per_sample(self, evaluation_output: Any, col: str, name: str) -> list[Measure]:
        """One Measure per question for a single ragas metric. Used to fill
        the dashboard TABLE with per-row drill-down. Dimensions identify
        which question + whether the judge LLM produced a score."""
        full_results = evaluation_output.get("full_results", [])
        aliases = {
            "response_relevancy": ["response_relevancy", "answer_relevancy"],
            "noise_sensitivity": ["noise_sensitivity"],
            "faithfulness": ["faithfulness"],
            "context_recall": ["context_recall"],
            "context_precision": ["context_precision"],
        }.get(col, [col])

        def resolve(row: dict):
            for k in row:
                for a in aliases:
                    if k == a or k.startswith(a + "("):
                        return row[k]
            return None

        measures = []
        for i, row in enumerate(full_results):
            qidx = i + 1
            question = str(row.get("user_input", f"Q{qidx}"))[:80]
            val = resolve(row)
            valid = val is not None and not (isinstance(val, float) and pd.isna(val))
            score = round(float(val), 4) if valid else 0.0
            err = None if valid else f"NaN: judge LLM did not score {col} for Q{qidx}"
            measures.append(Measure(
                name=name,
                score=score,
                description=f"Q{qidx}: {question}"[:250],
                error=err,
            ))
        return measures

    def _mean_metric(self, evaluation_output: Any, col: str, name: str) -> list[Measure]:
        summary = evaluation_output.get("summary", [])
        full_results = evaluation_output.get("full_results", [])
        if not summary:
            return [Measure(name=name, score=0.0)]
        # ragas may emit col names with parametric suffixes (e.g.
        # "noise_sensitivity(mode=relevant)") or aliased base names
        # ("answer_relevancy" for response_relevancy). Match by prefix
        # OR base aliases so all valid scores reach the dashboard.
        aliases = {
            "response_relevancy": ["response_relevancy", "answer_relevancy"],
            "noise_sensitivity": ["noise_sensitivity"],
            "faithfulness": ["faithfulness"],
            "context_recall": ["context_recall"],
            "context_precision": ["context_precision"],
        }.get(col, [col])

        def resolve(row: dict) -> Any:
            for k in row:
                for a in aliases:
                    if k == a or k.startswith(a + "("):
                        return row[k]
            return None

        # Count coverage so the description tells the user whether the
        # number is trustworthy (e.g. "12/20 questions scored").
        total = len(full_results)
        n_computed = 0
        for row in full_results:
            v = resolve(row)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                n_computed += 1

        measures = []
        for row in summary:
            val = resolve(row)
            valid = val is not None and not (isinstance(val, float) and pd.isna(val))
            score = float(val) if valid else 0.0
            coverage_note = f"({n_computed}/{total} scored)" if total else ""
            desc_mapping = {
                "faithfulness": "How factually grounded the answer is in the retrieved contexts. Higher is better (fewer hallucinations).",
                "context_recall": "Whether the retriever pulled in all the information needed to answer the question. Higher is better.",
                "context_precision": "Whether the contexts holding the relevant ground-truth info are ranked above the irrelevant ones. Higher is better.",
                "noise_sensitivity": "How easily irrelevant retrieved context throws the answer off. Lower is better (the inverse of the other four).",
                "response_relevancy": "How well the answer actually addresses the question. Higher is better.",
            }
            base_desc = desc_mapping.get(col, "Aggregate Average")
            desc = f"{base_desc} {coverage_note}" if coverage_note else base_desc
            err = None if valid else f"NaN: judge LLM did not score any sample for {col}"
            if not valid and coverage_note:
                err = f"{err} {coverage_note}"
            measures.append(Measure(
                name=name,
                score=round(score, 4),
                description=desc[:250],
                error=err,
            ))
        return measures


# =============================================================================
# === MODE 2 (commented out) — build RAG from PDFs, then evaluate =============
# =============================================================================
# Original Mode 2 implementation preserved here for future re-enable.
#
# To re-enable Mode 2 alongside Mode 1:
#   1. Uncomment the Mode 2 fields in form_schema.py
#   2. Add a second `@evaluation_input(name="pdf-files", required=False, ...)`
#      decorator on the class
#   3. In _run_evaluation: branch on whether pdf-files input is present.
#      If present -> Mode 2 (build RAG inline, generate answers, then evaluate)
#      Else       -> Mode 1 (current code)
#   4. Restore helpers _build_pipeline_llm, _generate_answers,
#      _generate_paraphrases_and_answers from git history (pre-2026-05-27).
#
# Mode 2 took: PDFs + dataset (Q + ground_truth only) + pipeline LLM + judge
# LLM. Built RAG inline via: PDFLoader -> Chunker -> Embedder(HF) -> FAISS
# -> Retriever -> LLM. Ran each Q through the built RAG, captured
# (Q, ctx, answer), called ragas.evaluate() on the captures.
#
# Use cases for Mode 2:
#   - Design exploration ("could a baseline RAG work for these docs?")
#   - LLM comparison (same RAG architecture, different LLMs)
#   - Teaching / demo (end-to-end pipeline without external setup)
#
# NOT suitable for: testing a customer's actual prod RAG, because the plugin's
# built RAG architecture (simple chunker + FAISS + stuff_documents prompt) is
# almost certainly different from prod (Pinecone, re-rankers, query rewriting,
# custom prompts, hybrid retrieval, etc.). Mode 2 scores reflect the baseline
# RAG the plugin builds, NOT the customer's real RAG.
#
# Original Mode 2 also supported: multilingual evaluation (source_language +
# target_languages list), paraphrase generation, context translation, all via
# the LLM Factory translate model. Those features re-emerge with Mode 2.
# =============================================================================
