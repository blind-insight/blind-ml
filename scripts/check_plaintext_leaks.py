"""Guardrail scanner: detect plaintext training leaks in blind-ml.

blind-ml models train on encrypted aggregate counts from Blind Insight (BI),
never on raw data rows. This script uses AST parsing to statically analyze
source files and flag code that uses plaintext data in encrypted code paths —
e.g. a model fit() method that accepts a DataFrame, or an encrypted training
function that loads local data instead of querying BI.

Designed to run in CI to block PRs that introduce plaintext training in
encrypted code paths. Exits 0 if clean, 1 if violations are found.

Usage:
    python scripts/check_plaintext_leaks.py --check models
    python scripts/check_plaintext_leaks.py --check functions
    python scripts/check_plaintext_leaks.py --all
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files containing model classes to scan (Guardrail 1).
# After planned restructure: models.py becomes models/ directory.
# When that happens, update this to glob models/*.py — the scan logic stays the same.
MODELS_FILES = [REPO_ROOT / "blind_ml" / "models.py"]

# Files containing encrypted training functions to scan (Guardrail 2).
# After planned restructure: these split into benchmarking/encrypted/.
# When that happens, update this to glob that directory — the naming-convention
# checks below can also simplify since the directory itself defines the zone.
ENCRYPTED_SCAN_FILES = [
    REPO_ROOT / "blind_ml" / "models.py",
    REPO_ROOT / "blind_ml" / "demo_helpers.py",
    REPO_ROOT / "blind_ml" / "healthcare.py",
]

# fit-prefixed methods that are allowed to use plaintext data.
# refine_irls is an IRLS weight-refinement step, not primary model training.
ALLOWLISTED_METHODS = {"refine_irls"}

# Functions that match encrypted prefixes but are explicitly allowed to use plaintext.
# Validation functions compare encrypted vs plaintext predictions (not training).
# Demo functions may decrypt for display purposes.
EXCLUDED_FUNCTIONS = {
    "run_test_validation",
    "run_bc_test_validation",
    "run_bc_full_validation",
    "run_realtime_demo",
    "run_bc_realtime_demo",
}

# Functions with these prefixes are in the "encrypted zone" — they must train
# only via BI aggregate queries, never on local DataFrames.
ENCRYPTED_PREFIXES = ("run_encrypted_", "run_bi_", "get_encrypted_", "get_bi_")

# Additional encrypted entry points that don't follow the prefix convention.
ENCRYPTED_NAMED = {
    "run_bc_training",
    "run_bc_conditional_queries",
    "run_bc_pairwise_queries",
    "get_bc_base_rates",
}

# Patterns that indicate plaintext training inside fit* method bodies
# (Guardrail 1). Covers DataFrame operations, sklearn usage, and direct
# data-loading calls. Parentheses/brackets avoid partial matches.
FIT_BODY_INDICATORS = {
    "pd.DataFrame",
    "pd.get_dummies(",
    "pd.to_numeric(",
    "pd.crosstab(",
    ".iterrows()",
    "df[",
    "df_local[",
    "sklearn",
    "read_csv(",
    "read_sql",
    "read_parquet(",
}


# Patterns forbidden inside encrypted function bodies (Guardrail 2).
# Includes data loaders, plaintext training calls, DataFrame operations,
# and sklearn usage. Any match means the function uses local data
# instead of BI encrypted aggregate queries.
ENCRYPTED_BODY_FORBIDDEN = {
    "load_data(",
    "load_training_data(",
    "load_test_data(",
    "train_plaintext_",
    ".fit(df",
    ".fit(df_local",
    ".fit(X,",
    ".fit(X_encoded",
    ".fit(features",
    ".fit(matrix",
    "pd.get_dummies(",
    "pd.to_numeric(",
    "pd.crosstab(",
    "pd.read_sql",
    "pd.read_csv(",
    "read_csv(",
    "sklearn",
}

# Parameter names that indicate a function receives local data.
# An encrypted function should receive BI client/org/dataset/schema,
# not raw arrays or DataFrames.
ENCRYPTED_PARAM_FORBIDDEN = {"df", "df_local", "df_train", "X_train", "X", "y", "Y", "features", "labels"}


def _is_encrypted_function(name: str) -> bool:
    """Return True if this function name is in the encrypted zone and should be scanned."""
    if name.startswith("train_plaintext_"):
        return False
    if name in EXCLUDED_FUNCTIONS:
        return False
    if name in ENCRYPTED_NAMED:
        return True
    return any(name.startswith(p) for p in ENCRYPTED_PREFIXES)


def _body_source(node: ast.FunctionDef, source_lines: list[str]) -> str:
    """Extract the source text of a function body, excluding docstrings."""
    if not node.body:
        return ""
    stmts = node.body
    if (
        isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, (ast.Constant, ast.JoinedStr))
        and isinstance(getattr(stmts[0].value, "value", None), str)
    ):
        stmts = stmts[1:]
    if not stmts:
        return ""
    start = stmts[0].lineno - 1
    end = node.end_lineno or (start + 1)
    return "\n".join(source_lines[start:end])


def check_models(paths: list[Path] | None = None) -> list[str]:
    """Guardrail 1: scan model classes for fit* methods that use plaintext training."""
    violations: list[str] = []
    targets = paths or MODELS_FILES

    for fpath in targets:
        if not fpath.exists():
            continue

        source = fpath.read_text()
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(fpath))
        rel = fpath.relative_to(REPO_ROOT)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                name = item.name

                if not name.startswith("fit"):
                    continue
                if name in ALLOWLISTED_METHODS:
                    continue

                body_text = _body_source(item, source_lines)

                has_df_param = any(
                    arg.arg in ("df", "df_local", "dataframe") for arg in item.args.args if arg.arg != "self"
                )

                has_df_annotation = False
                for arg in item.args.args:
                    if arg.annotation and "DataFrame" in ast.dump(arg.annotation):
                        has_df_annotation = True
                        break

                has_df_body = any(indicator in body_text for indicator in FIT_BODY_INDICATORS)

                # Check three ways: parameter names, type annotations, and body operations.
                # Any single match is enough to flag a violation.
                if has_df_param or has_df_annotation or has_df_body:
                    violations.append(f"{rel}:{item.lineno}  {node.name}.{name}() uses plaintext training")

    return violations


def _body_forbidden(node: ast.FunctionDef, source_lines: list[str]) -> list[str]:
    """Return ENCRYPTED_BODY_FORBIDDEN patterns found in a function body."""
    body_text = _body_source(node, source_lines)
    return [ind for ind in ENCRYPTED_BODY_FORBIDDEN if ind in body_text]


def _call_targets(node: ast.FunctionDef) -> list[str]:
    """Return names of bare function calls (not method calls) in a function body."""
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            names.append(child.func.id)
    return names


def check_functions(paths: list[Path] | None = None) -> list[str]:
    """Guardrail 2: scan encrypted functions for plaintext training references."""
    violations: list[str] = []
    targets = paths or ENCRYPTED_SCAN_FILES

    for fpath in targets:
        if not fpath.exists():
            continue

        source = fpath.read_text()
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(fpath))
        rel = fpath.relative_to(REPO_ROOT)

        all_funcs: dict[str, ast.FunctionDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                all_funcs[node.name] = node

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_encrypted_function(node.name):
                continue

            reasons: list[str] = []

            bad_params = [arg.arg for arg in node.args.args if arg.arg in ENCRYPTED_PARAM_FORBIDDEN]
            if bad_params:
                reasons.append(f"param {', '.join(bad_params)}")

            reasons.extend(_body_forbidden(node, source_lines))

            for callee_name in _call_targets(node):
                if callee_name in all_funcs and not _is_encrypted_function(callee_name):
                    callee_hits = _body_forbidden(all_funcs[callee_name], source_lines)
                    for hit in callee_hits:
                        reasons.append(f"{hit} (via {callee_name})")

            if reasons:
                detail = "; ".join(reasons)
                violations.append(f"{rel}:{node.lineno}  {node.name}() references plaintext training: {detail}")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Plaintext leak guardrails")
    parser.add_argument(
        "--check",
        choices=["models", "functions", "imports", "callgraph", "all"],
        default="all",
    )
    parser.add_argument("--all", action="store_true", help="Alias for --check all")
    args = parser.parse_args()

    if args.all:
        args.check = "all"

    violations: list[str] = []

    if args.check in ("models", "all"):
        violations.extend(check_models())

    if args.check in ("functions", "all"):
        violations.extend(check_functions())

    if not violations:
        print("OK — no plaintext training leaks detected.")
        return 0

    print(f"FAIL — {len(violations)} violation(s) found:\n")
    for v in violations:
        print(f"  {v}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
