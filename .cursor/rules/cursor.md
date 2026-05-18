---
description: Project-wide conventions for blind-ml
alwaysApply: true
---

# blind-ml Project Rules

This repo demonstrates training sklearn-style ML models on encrypted data via Blind Insight's searchable encryption. The two demo notebooks live at the repo root; shared code is in `blind_ml/`.

## Before Any Task

- Skim `APPROACH.md` — explains how ML on encrypted data works (the algorithms train on aggregate counts, not raw records).
- For ML work, consult `.cursor/skills/ml-expert/SKILL.md` and `debugging-patterns.md`.
- For BI-specific questions, read the [official Blind Insight docs](https://docs.blindinsight.io).

If nothing specific applies, say "Reviewed skills — no specific constraints apply."

## Notebook Work

When editing `.ipynb` files or the helpers in `blind_ml/`:

- Use `pd.get_dummies()` not `LabelEncoder` for one-hot encoding (sklearn ordinal labels produce different splits than BI's categorical handling).
- Keep all data paths relative to repo root (`demo_data/...`) — notebooks run from the repo root by convention.
- Cell outputs in committed notebooks should be the canonical demo run on the 50K / 20K sample data shipped in `demo_data/upload_batches/`.

## Linting

Run before committing:
- `ruff check --fix` then `ruff format` on any `.py` you touched
- `nbqa ruff --fix` for `.ipynb` (optional but recommended)

## Sensitive Data

- Never commit `.env` (it's in `.gitignore`). Use `.env.example` as the template.
- Never commit anything from `demo_data/plaintext/` (also gitignored).
- The four committed sample batches in `demo_data/upload_batches/` are synthetic — they contain no real PII or financial data.
