# Contributing to blind-ml

Thanks for your interest in expanding the library of ML-on-encrypted-data demos. Contributions of all sizes are welcome — bug fixes, new algorithm demos, dataset examples, performance work, documentation improvements.

## Ground rules

- **Fork → branch → PR.** External contributors should fork the repo, push to a branch on their fork, and open a PR against `main`.
- **One change per PR.** Smaller PRs ship faster. If you're adding a new notebook, that's its own PR; don't bundle it with infrastructure changes.
- **Open an issue first for anything substantial.** Sketches and prototypes are great — getting alignment on the approach before you invest a weekend on a notebook saves everyone time.
- **All contributions are MIT-licensed.** By submitting a PR, you agree your contribution is licensed under the same MIT license as the rest of the repo.

## Setup

```bash
git clone https://github.com/<your-fork>/blind-ml.git
cd blind-ml
pip install -e ".[dev,notebooks]"   # editable install + dev + notebook deps
cp .env.example .env                 # fill in your Blind Insight credentials
```

You'll need a Blind Insight account to run the demos end-to-end. Sign up at [blindinsight.com](https://blindinsight.com); the proxy and CLI are downloadable from [docs.blindinsight.io](https://docs.blindinsight.io).

### Demo data

Upload batches and SQLite files are **not** in this repo (keeps clones small). Either:

1. **Generate** — `python3 scripts/generate_fraud_data.py` and/or `python3 scripts/generate_healthcare_data.py`, or
2. **Download** — copy JSON from [demo-datasets `datasets/blind-ml/`](https://github.com/blind-insight/demo-datasets/tree/main/datasets/blind-ml) into `demo_data/upload_batches/` (see [demo_data/README.md](demo_data/README.md)).

Cursor users: see [.cursor/rules/cursor.md](.cursor/rules/cursor.md) and [.cursor/skills/blind-ml-demo/SKILL.md](.cursor/skills/blind-ml-demo/SKILL.md) for notebook and helper conventions.

## Linting and formatting

Run before opening a PR — CI will reject otherwise.

```bash
ruff check --fix .
ruff format .
```

For notebooks (same ignores as CI):

```bash
nbqa ruff --fix --extend-ignore=E402,F401,E702,E401,I001,F811,F541 breast_cancer.ipynb fraud.ipynb
```

```bash
python3 scripts/smoke_test.py
```

Clear notebook outputs before pushing:

```bash
jupyter nbconvert --clear-output --inplace fraud.ipynb breast_cancer.ipynb
```

## Submitting a PR

1. Push your branch to your fork
2. Open a PR against `blind-insight/blind-ml:main`
3. Fill out the PR template (what changed, how you tested, linked issue)
4. CI runs automatically. A Blind Insight maintainer will review and merge.

External contributors cannot self-merge — every PR is reviewed and merged by a maintainer.

## What we're especially looking for

- **New algorithm demos** — pick any algorithm from the [supported list in APPROACH.md](APPROACH.md#what-algorithms-work-on-encrypted-data) and build a notebook
- **New domains / use cases** — insurance fraud, identity verification, healthcare beyond BC, etc.
- **Performance work** — query batching, smarter caching, parallel execution patterns
- **Docs** — clearer explanations, diagrams, runnable tutorials

## Questions?

Open a discussion or ping the maintainers in your PR. We're happy to help shape contributions early.
