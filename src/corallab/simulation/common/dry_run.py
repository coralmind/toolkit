"""DryRunCollector — accumulates LLM call records during --dry-run and renders a summary."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class DryRunRecord:
    cache_key: str | None
    trace_role: str
    prompt_tokens: int


class DryRunCollector:
    """Accumulates dry-run call records and renders a token/cache summary."""

    def __init__(self) -> None:
        self._records: list[DryRunRecord] = []

    def record(
        self,
        *,
        cache_key: str | None,
        trace_role: str,
        prompt_tokens: int,
        provider_model: str,  # accepted for forward-compat but not printed
    ) -> None:
        del provider_model
        self._records.append(
            DryRunRecord(
                cache_key=cache_key,
                trace_role=trace_role,
                prompt_tokens=prompt_tokens,
            )
        )

    def render_summary(self) -> str:
        """Build per-cache-key breakdown and totals as a multi-line string."""
        if not self._records:
            return "\nDRY RUN — no LLM calls recorded\n"

        groups: dict[str, list[DryRunRecord]] = defaultdict(list)
        for rec in self._records:
            key = rec.cache_key if rec.cache_key is not None else f"(no-key) {rec.trace_role}"
            groups[key].append(rec)

        rows: list[tuple[str, int, int, int, float]] = []
        for label, recs in groups.items():
            n_calls = len(recs)
            total_input_tok = sum(r.prompt_tokens for r in recs)
            # stable_tokens = shortest prompt across calls. This is an upper bound on the
            # shared cacheable prefix (the prefix can't exceed the shortest prompt), so the
            # cache-savings projection may over-estimate savings / under-estimate cost.
            stable_tok = min(r.prompt_tokens for r in recs)
            hits = max(0, n_calls - 1)
            hit_rate = hits / n_calls if n_calls > 0 else 0.0
            rows.append((label, n_calls, total_input_tok, stable_tok, hit_rate))

        col_label = max(max(len(r[0]) for r in rows), len("Cache Key"))

        header = (
            f"{'Cache Key':<{col_label}}  {'Calls':>5}  {'Input tok':>10}  "
            f"{'Stable tok':>10}  {'Cache proj':>10}"
        )
        sep = "─" * len(header)

        lines = [
            "",
            "DRY RUN — no API calls made",
            "",
            header,
            sep,
        ]

        total_calls = 0
        total_input_tok = 0
        total_hits = 0

        for label, n_calls, input_tok, stable_tok, hit_rate in rows:
            lines.append(
                f"{label:<{col_label}}  {n_calls:>5}  {input_tok:>10,}  "
                f"{stable_tok:>10,}  {hit_rate:>9.1%}"
            )
            total_calls += n_calls
            total_input_tok += input_tok
            total_hits += max(0, n_calls - 1)

        overall_hit_rate = total_hits / total_calls if total_calls > 0 else 0.0

        lines.extend(
            [
                sep,
                (
                    f"{'TOTAL':<{col_label}}  {total_calls:>5}  {total_input_tok:>10,}  "
                    f"{'':>10}  {overall_hit_rate:>9.1%}"
                ),
                "",
            ]
        )
        return "\n".join(lines)
