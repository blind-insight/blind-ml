"""
Generate synthetic breast cancer screening data for the Blind Insight healthcare demo.

Outputs:
  - 5 JSON training batches (20K records each = 100K total)
  - 1 JSON test batch (10,000 records)
  - SQLite databases: bc_train.db (100K) and bc_test.db (10K)

Cancer outcomes correlate with risk factors at published epidemiological
relative risk levels (Gail model / SEER), so the BCRAT comparison in the
notebook produces realistic agreement rates.

Each 20K batch has the same feature distributions, so the demo works with
1 batch (20K) or all 5 (100K).

Usage:
  python scripts/generate_healthcare_data.py
  python scripts/generate_healthcare_data.py --json-only
  python scripts/generate_healthcare_data.py --sqlite-only
  python scripts/generate_healthcare_data.py --append
  python scripts/generate_healthcare_data.py --schema-url https://app.blindinsight.io/...
"""

import argparse
import json
import math
import os
import random
import sqlite3
import sys
from pathlib import Path

TRAIN_BATCHES = 5
TRAIN_BATCH_SIZE = 20_000
TEST_RECORDS = 10_000
BASE_SEED = 99
# Seed offset for expanded batches — ensures new batches never replay
# the same RNG sequence as the original 50K (which used offsets 0–4).
# Each new round of expansion should use a new offset tier (50, 100, ...).
BATCH_SEED_OFFSET = 50

JSON_OUTPUT_DIR = Path("demo_data/upload_batches")
TRAIN_DB_PATH = Path("demo_data/plaintext/bc_train.db")
TEST_DB_PATH = Path("demo_data/plaintext/bc_test.db")

SCHEMA_PLACEHOLDER = "REPLACE_WITH_YOUR_SCHEMA_URL"


def get_schema_url(override=None):
    """Return schema URL from explicit override, env vars, or placeholder fallback."""
    if override:
        return override
    explicit = os.getenv("BI_SCHEMA_URL")
    if explicit:
        return explicit
    org = os.getenv("BI_ORG", "")
    dataset = os.getenv("BI_DATASET", "breast-screening-data")
    schema = os.getenv("BI_SCHEMA", "training")
    if org:
        return f"https://app.blindinsight.io/api/v1/orgs/{org}/datasets/{dataset}/schemas/{schema}/"
    return SCHEMA_PLACEHOLDER


# ============================================================================
# Feature distributions (based on SEER, Paige et al. 2023, CDC)
# ============================================================================

AGE_WEIGHTS = {
    "40_49": 0.25,
    "50_59": 0.30,
    "60_69": 0.28,
    "70_74": 0.17,
}
AGE_RANGES = {
    "40_49": (40, 49),
    "50_59": (50, 59),
    "60_69": (60, 69),
    "70_74": (70, 74),
}

RACE_WEIGHTS = {
    "white": 0.55,
    "hispanic": 0.20,
    "asian_pi": 0.12,
    "black": 0.08,
    "other": 0.05,
}

MENARCHE_WEIGHTS = {
    "under_12": 0.30,
    "12_13": 0.45,
    "14_plus": 0.25,
}
MENARCHE_RANGES = {
    "under_12": (8, 11),
    "12_13": (12, 13),
    "14_plus": (14, 17),
}

FIRST_BIRTH_WEIGHTS = {
    "nulliparous": 0.15,
    "under_20": 0.12,
    "20_24": 0.25,
    "25_29": 0.28,
    "30_plus": 0.20,
}
FIRST_BIRTH_RANGES = {
    "nulliparous": (0, 0),
    "under_20": (15, 19),
    "20_24": (20, 24),
    "25_29": (25, 29),
    "30_plus": (30, 45),
}

FAMILY_HISTORY_RATE = 0.15
RELATIVES_IF_POSITIVE = {1: 0.85, 2: 0.12, 3: 0.03}

BIOPSY_RATE = 0.15
BIOPSY_COUNT_IF_POSITIVE = {1: 0.75, 2: 0.18, 3: 0.05, 4: 0.015, 5: 0.005}

ATYPICAL_RATE_IF_BIOPSY = 0.05

DENSITY_WEIGHTS = {1: 0.10, 2: 0.40, 3: 0.40, 4: 0.10}

# ============================================================================
# BCRAT-inspired relative risks for cancer outcome generation
# ============================================================================

# Age-specific baseline 5-year hazard (attributable-risk-adjusted SEER rates).
# Calibrated to produce ~3.3% overall cancer rate (matches Paige et al. 2023 cohort).
BASELINE_5YR = {
    "40_49": 0.008,
    "50_59": 0.015,
    "60_69": 0.022,
    "70_74": 0.028,
}

# Gail model relative risks (Z-column from Costantino/Gail validation)
RR_MENARCHE = {"under_12": 1.207, "12_13": 1.098, "14_plus": 1.000}


def rr_biopsy(num_biopsies, age):
    if num_biopsies == 0:
        return 1.000
    if age < 50:
        return 1.683 if num_biopsies == 1 else 2.750
    return 1.237 if num_biopsies == 1 else 1.539


def rr_birth_relatives(first_birth_cat, num_relatives):
    table = {
        "nulliparous": {0: 1.000, 1: 2.560, 2: 6.168},
        "under_20": {0: 1.000, 1: 2.560, 2: 6.168},
        "20_24": {0: 1.240, 1: 2.640, 2: 5.318},
        "25_29": {0: 1.550, 1: 2.705, 2: 4.591},
        "30_plus": {0: 1.930, 1: 2.779, 2: 3.953},
    }
    capped = min(num_relatives, 2)
    return table.get(first_birth_cat, table["under_20"]).get(capped, 1.0)


RR_ATYPICAL = {"yes": 1.82, "no": 0.93, "unknown": 1.00}
RR_DENSITY = {1: 0.70, 2: 0.90, 3: 1.15, 4: 1.60}

# ============================================================================
# Record generation
# ============================================================================

INT_COLS = {
    "age",
    "age_at_menarche",
    "age_at_first_birth",
    "num_first_degree_relatives",
    "num_prior_biopsies",
    "breast_density",
    "cancer_5yr",
}
ALL_COLS = [
    "patient_id",
    "age",
    "age_group",
    "age_at_menarche",
    "menarche_category",
    "age_at_first_birth",
    "num_first_degree_relatives",
    "family_history",
    "num_prior_biopsies",
    "has_prior_biopsy",
    "atypical_hyperplasia",
    "race_ethnicity",
    "breast_density",
    "cancer_5yr",
]


def weighted_choice(rng, weights_dict):
    items = list(weights_dict.keys())
    weights = list(weights_dict.values())
    return rng.choices(items, weights=weights, k=1)[0]


def generate_record(rng, patient_id):
    age_group = weighted_choice(rng, AGE_WEIGHTS)
    lo, hi = AGE_RANGES[age_group]
    age = rng.randint(lo, hi)

    race = weighted_choice(rng, RACE_WEIGHTS)

    menarche_cat = weighted_choice(rng, MENARCHE_WEIGHTS)
    m_lo, m_hi = MENARCHE_RANGES[menarche_cat]
    age_at_menarche = rng.randint(m_lo, m_hi)

    first_birth_cat = weighted_choice(rng, FIRST_BIRTH_WEIGHTS)
    fb_lo, fb_hi = FIRST_BIRTH_RANGES[first_birth_cat]
    age_at_first_birth = rng.randint(fb_lo, fb_hi) if fb_hi > 0 else 0

    has_family = rng.random() < FAMILY_HISTORY_RATE
    if has_family:
        num_relatives = int(weighted_choice(rng, RELATIVES_IF_POSITIVE))
    else:
        num_relatives = 0
    family_history = "yes" if has_family else "no"

    has_biopsy = rng.random() < BIOPSY_RATE
    if has_biopsy:
        num_biopsies = int(weighted_choice(rng, BIOPSY_COUNT_IF_POSITIVE))
    else:
        num_biopsies = 0
    has_prior_biopsy = "yes" if has_biopsy else "no"

    if has_biopsy:
        if rng.random() < ATYPICAL_RATE_IF_BIOPSY:
            atypical = "yes"
        else:
            atypical = "no"
    else:
        atypical = "unknown"

    density = int(weighted_choice(rng, DENSITY_WEIGHTS))

    # --- Cancer outcome via BCRAT-like relative risk model ---
    composite_rr = (
        RR_MENARCHE[menarche_cat]
        * rr_biopsy(num_biopsies, age)
        * rr_birth_relatives(first_birth_cat, num_relatives)
        * RR_ATYPICAL[atypical]
        * RR_DENSITY[density]
    )
    baseline = BASELINE_5YR[age_group]
    prob_cancer = 1.0 - math.pow(1.0 - baseline, composite_rr)
    cancer_5yr = 1 if rng.random() < prob_cancer else 0

    return {
        "patient_id": str(patient_id),
        "age": age,
        "age_group": age_group,
        "age_at_menarche": age_at_menarche,
        "menarche_category": menarche_cat,
        "age_at_first_birth": age_at_first_birth,
        "num_first_degree_relatives": num_relatives,
        "family_history": family_history,
        "num_prior_biopsies": num_biopsies,
        "has_prior_biopsy": has_prior_biopsy,
        "atypical_hyperplasia": atypical,
        "race_ethnicity": race,
        "breast_density": density,
        "cancer_5yr": cancer_5yr,
    }


def generate_batch(batch_seed, count, start_id):
    rng = random.Random(batch_seed)
    return [generate_record(rng, start_id + i) for i in range(count)]


def write_json_batch(records, filepath, schema_url):
    json_records = [{"data": r, "schema": schema_url} for r in records]
    with open(filepath, "w") as f:
        json.dump(json_records, f)
    return os.path.getsize(filepath)


def write_sqlite(records, db_path, table_name):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    col_defs = ", ".join(f'"{c}" INTEGER' if c in INT_COLS else f'"{c}" TEXT' for c in ALL_COLS)
    cur.execute(f"CREATE TABLE {table_name} ({col_defs})")

    placeholders = ", ".join("?" for _ in ALL_COLS)
    insert_sql = f"INSERT INTO {table_name} ({', '.join(ALL_COLS)}) VALUES ({placeholders})"

    batch_size = 5_000
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        rows = []
        for rec in batch:
            row = tuple(int(rec[c]) if c in INT_COLS else str(rec[c]) for c in ALL_COLS)
            rows.append(row)
        cur.executemany(insert_sql, rows)
        conn.commit()

    conn.close()


def append_to_sqlite(records, db_path, table_name):
    """Append records to an existing SQLite table (creates table+DB if absent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    col_defs = ", ".join(f'"{c}" INTEGER' if c in INT_COLS else f'"{c}" TEXT' for c in ALL_COLS)
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")

    placeholders = ", ".join("?" for _ in ALL_COLS)
    insert_sql = f"INSERT INTO {table_name} ({', '.join(ALL_COLS)}) VALUES ({placeholders})"

    batch_size = 5_000
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        rows = [tuple(int(rec[c]) if c in INT_COLS else str(rec[c]) for c in ALL_COLS) for rec in batch]
        cur.executemany(insert_sql, rows)
        conn.commit()

    conn.close()


def _table_exists(cur, table_name):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def get_max_patient_id(db_path, table_name):
    """Return the highest patient_id currently in the table, or 0 if empty/absent."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    if not _table_exists(cur, table_name):
        conn.close()
        return 0
    cur.execute(f"SELECT MAX(CAST(patient_id AS INTEGER)) FROM {table_name}")
    result = cur.fetchone()[0]
    conn.close()
    return result or 0


def verify_distributions(all_batches):
    print("\nPer-batch distribution verification:")
    print(f"  {'Batch':<12} {'Records':>8} {'Cancer%':>9} {'NoCanc%':>9}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 9} {'-' * 9}")
    for label, records in all_batches:
        cancer = sum(1 for r in records if r["cancer_5yr"] == 1)
        no_cancer = len(records) - cancer
        pct_cancer = cancer / len(records) * 100
        pct_no = no_cancer / len(records) * 100
        print(f"  {label:<12} {len(records):>8,} {pct_cancer:>8.1f}% {pct_no:>8.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate breast cancer screening demo data for Blind Insight")
    parser.add_argument("--json-only", action="store_true", help="Write JSON batches only; skip SQLite")
    parser.add_argument("--sqlite-only", action="store_true", help="Write SQLite only; skip JSON batches")
    parser.add_argument(
        "--append", action="store_true", help="Append new records to existing bc_train.db instead of recreating it"
    )
    parser.add_argument("--schema-url", default=None, help="Override schema URL written into JSON records")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    total_train = TRAIN_BATCHES * TRAIN_BATCH_SIZE
    schema_url = get_schema_url(args.schema_url)

    # In append mode, determine how many records already exist so IDs don't collide
    existing_count = 0
    if args.append:
        existing_count = get_max_patient_id(TRAIN_DB_PATH, "train")
        print(f"Append mode: {existing_count:,} records already in {TRAIN_DB_PATH}")

    print(f"Generating {total_train:,} training records ({TRAIN_BATCHES} batches x {TRAIN_BATCH_SIZE:,})...")
    print(f"Generating {TEST_RECORDS:,} test records (1 batch)...")
    print(f"Schema URL: {schema_url}")
    print("Each batch seeded independently for consistent distributions.\n")

    train_batches = []
    all_train_records = []
    for batch_idx in range(TRAIN_BATCHES):
        batch_seed = BASE_SEED + BATCH_SEED_OFFSET + batch_idx
        start_id = existing_count + batch_idx * TRAIN_BATCH_SIZE + 1
        records = generate_batch(batch_seed, TRAIN_BATCH_SIZE, start_id)
        train_batches.append((f"bc_train_{batch_idx + 1:02d}", records))
        all_train_records.extend(records)

    test_seed = BASE_SEED + 100
    test_start_id = (existing_count + total_train) + 1
    test_records = generate_batch(test_seed, TEST_RECORDS, test_start_id)

    verify_distributions(train_batches + [("test", test_records)])

    if not args.sqlite_only:
        JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting JSON batches to {JSON_OUTPUT_DIR}/...")
        for batch_idx, (label, records) in enumerate(train_batches, 1):
            filepath = JSON_OUTPUT_DIR / f"bc_train_batch_{batch_idx:02d}.json"
            size = write_json_batch(records, filepath, schema_url)
            print(f"  {filepath.name}: {len(records):,} records ({size / 1024 / 1024:.1f} MB)")

        test_path = JSON_OUTPUT_DIR / "bc_test_batch_01.json"
        size = write_json_batch(test_records, test_path, schema_url)
        print(f"  {test_path.name}: {len(test_records):,} records ({size / 1024 / 1024:.1f} MB)")

    if not args.json_only:
        if args.append:
            print(f"\nAppending {total_train:,} records to {TRAIN_DB_PATH}...")
            append_to_sqlite(all_train_records, TRAIN_DB_PATH, "train")
        else:
            print(f"\nBuilding {TRAIN_DB_PATH}...")
            write_sqlite(all_train_records, TRAIN_DB_PATH, "train")
        print(f"  Done: {total_train:,} records written")

        print(f"Building {TEST_DB_PATH}...")
        write_sqlite(test_records, TEST_DB_PATH, "test")
        print(f"  Done: {TEST_RECORDS:,} records")

    if not args.json_only:
        expected_train = existing_count + total_train if args.append else total_train
        print("\nSQLite verification:")
        for db_path, table, expected in [
            (TRAIN_DB_PATH, "train", expected_train),
            (TEST_DB_PATH, "test", TEST_RECORDS),
        ]:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE cancer_5yr = 1")
            cancer = cur.fetchone()[0]
            conn.close()
            pct = cancer / count * 100
            status = "OK" if count == expected else "MISMATCH"
            print(f"  {db_path}: {count:,} records, {pct:.1f}% cancer [{status}]")


if __name__ == "__main__":
    main()
