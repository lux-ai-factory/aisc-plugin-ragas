from __future__ import annotations

from typing import Any

from sqlalchemy import Float, cast, func, select
from sqlalchemy.orm import Session

from resources.sql_alchemy import Measurement, Metric, Observation

PER_RUN_METRICS = (
    "MLA RAGAS Run Success",
    "RAGAS Overall Score",
    "RAGAS Faithfulness",
    "RAGAS Context Recall",
    "RAGAS Context Precision",
    "RAGAS Noise Sensitivity",
    "RAGAS Response Relevancy",
)

class MLARagasDataLoader:
    def __init__(self, db_session: Session | None = None):
        if not db_session:
            raise RuntimeError("No database session available")
        self.session = db_session

    def _scores_by_metric(self) -> dict[str, list[float]]:
        stmt = (
            select(
                Metric.name.label("metric_name"),
                cast(Measurement.score, Float).label("score"),
            )
            .join(Measurement, Measurement.metric_id == Metric.id)
            .join(Observation, Measurement.observation_id == Observation.id)
            .where(Metric.name.in_(PER_RUN_METRICS))
        )
        rows = self.session.execute(stmt).mappings().all()
        out: dict[str, list[float]] = {name: [] for name in PER_RUN_METRICS}
        for row in rows:
            v = row["score"]
            if v is None:
                continue
            out.setdefault(row["metric_name"], []).append(float(v))
        return out

    def _avg(self, vals: list[float]) -> float | None:
        return (sum(vals) / len(vals)) if vals else None

    def _n_observations(self) -> int:
        stmt = (
            select(func.count(func.distinct(Observation.id)))
            .join(Measurement, Measurement.observation_id == Observation.id)
            .join(Metric, Measurement.metric_id == Metric.id)
            .where(Metric.name.in_(PER_RUN_METRICS))
        )
        return int(self.session.scalar(stmt) or 0)

    def compute_statistics(self) -> dict[str, Any]:
        scores = self._scores_by_metric()
        n_obs = self._n_observations()

        rates = {
            "overall_score": self._avg(scores.get("RAGAS Overall Score", [])),
            "faithfulness": self._avg(scores.get("RAGAS Faithfulness", [])),
            "context_recall": self._avg(scores.get("RAGAS Context Recall", [])),
            "context_precision": self._avg(scores.get("RAGAS Context Precision", [])),
            "noise_sensitivity": self._avg(scores.get("RAGAS Noise Sensitivity", [])),
            "response_relevancy": self._avg(scores.get("RAGAS Response Relevancy", [])),
            "success_rate": self._avg(scores.get("MLA RAGAS Run Success", [])),
        }

        counts = {
            "n_runs": len(scores.get("RAGAS Overall Score", [])) or None,
        }

        units = {
            "overall_score": "score",
            "faithfulness": "score",
            "context_recall": "score",
            "context_precision": "score",
            "noise_sensitivity": "score",
            "response_relevancy": "score",
            "success_rate": "ratio",
            "n_runs": "runs",
        }

        metrics_table = []
        for name, value in {**rates, **counts}.items():
            metrics_table.append({
                "name": name,
                "value": value,
                "unit": units.get(name, ""),
                "n_observations": n_obs,
            })

        return {
            "nObservations": n_obs,
            "summary": {
                "evaluations": n_obs,
                "total_runs": int(counts["n_runs"]) if counts["n_runs"] else None,
                **rates,
            },
            "rates": rates,
            "counts": counts,
            "metricsTable": metrics_table,
        }
