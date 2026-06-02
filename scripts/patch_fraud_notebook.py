"""
One-shot patch for fraud.ipynb to align with the fix branch:

  Cell [6] (NB):   rename run_bi_training kwargs n_high=, n_low= → n_high_local=, n_low_local=
                    (run_bi_training now fetches base rates from BI internally)
  Cell [8] (DT):   replace `raw_results = build_raw_results_local(df, feature_values)`
                    with    `raw_results = bi["raw_results"]`  (use real encrypted counts)
  Cell [10] (LR):  no change needed — uses `raw_results` from global scope, now BI-sourced.

Run from the repo root: `venv/bin/python scripts/patch_fraud_notebook.py`
Idempotent — safe to re-run.
"""

import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "fraud.ipynb"


def patch_cell_source(src: str, replacements: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Apply each (old, new) replacement to `src`. Returns (new_src, applied_or_already)."""
    log = []
    for old, new in replacements:
        if new in src and old not in src:
            log.append(f"  already applied: {old.splitlines()[0]!r}")
            continue
        if old not in src:
            log.append(f"  NOT FOUND: {old.splitlines()[0]!r}")
            continue
        src = src.replace(old, new)
        log.append(f"  patched: {old.splitlines()[0]!r}")
    return src, log


def main() -> int:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    patches = {
        6: [
            (  # NB cell
                "bi = run_bi_training(\n"
                "    client, ORG, DATASET, SCHEMA, feature_values,\n"
                "    n_high=n_high_local, n_low=n_low_local,\n"
                ")",
                "bi = run_bi_training(\n"
                "    client, ORG, DATASET, SCHEMA, feature_values,\n"
                "    n_high_local=n_high_local, n_low_local=n_low_local,\n"
                ")",
            )
        ],
        8: [
            (  # DT cell
                "# Build marginal counts from local mirror (verified 100% match with BI).\n"
                "# Using local ensures consistency with the local cross-tabs/IRLS steps below.\n"
                "raw_results = build_raw_results_local(df, feature_values)",
                "# Reuse the encrypted aggregate counts NB already fetched from Blind Insight.\n"
                "# DT's root split is chosen from these BI counts; deeper splits use local\n"
                '# cross-tabs (see APPROACH.md — "root from BI marginals, deeper from local").\n'
                "# LR consumes the same raw_results downstream — zero additional BI queries.\n"
                'raw_results = bi["raw_results"]',
            )
        ],
    }

    any_changes = False
    for cell_idx, replacements in patches.items():
        cell = cells[cell_idx]
        if cell["cell_type"] != "code":
            print(f"cell [{cell_idx}] is not code — skipping", file=sys.stderr)
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        new_src, log = patch_cell_source(src, replacements)
        print(f"cell [{cell_idx}]:")
        for line in log:
            print(line)
        if new_src != src:
            any_changes = True
            # ipynb format: source is a list of lines, each line ends with \n except possibly the last
            lines = new_src.splitlines(keepends=True)
            cell["source"] = lines

    if any_changes:
        NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"\nwrote {NB_PATH}")
    else:
        print("\nno changes needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
