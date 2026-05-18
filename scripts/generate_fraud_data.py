"""
Generate demo data for the Blind Insight ML notebook.

Outputs (default run):
  - 10 JSON training batches (50K records each = 500K total)
  - 1 JSON test batch (50,000 records)
  - SQLite databases: fraud_train.db (500K) and fraud_test.db (50K)

Outputs (--append-noise):
  - 2 JSON noise training batches (50K each = 100K total, fraud_type flipped)
  - 1 JSON noise test batch (4,000 records, fraud_type flipped)
  - Appends noise records to existing SQLite databases

Each 50K batch (train and test) has the same feature distributions (~65% high
risk, fraud_type correlates with risk class). This means the demo works
correctly whether the user uploads 1 training batch (50K) or all 10 (500K).
The SQLite DB is written in batch order so LIMIT N*50000 always gives a
representative slice.

Usage:
  python scripts/generate_fraud_data.py                  # full generation (500K + 50K)
  python scripts/generate_fraud_data.py --json-only      # JSON batches only
  python scripts/generate_fraud_data.py --sqlite-only    # SQLite DBs only
  python scripts/generate_fraud_data.py --append-noise   # add 100K+4K noisy records
"""

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

TRAIN_BATCHES = 10
TRAIN_BATCH_SIZE = 50_000
TEST_RECORDS = 50_000
NOISE_TRAIN = 100_000
NOISE_TEST = 4_000
BASE_SEED = 42
NOISE_SEED = 9999

JSON_OUTPUT_DIR = Path("demo_data/upload_batches")
TRAIN_DB_PATH = Path("demo_data/plaintext/fraud_train.db")
TEST_DB_PATH = Path("demo_data/plaintext/fraud_test.db")

SCHEMA_PLACEHOLDER = "REPLACE_WITH_YOUR_SCHEMA_URL"

# === Distributions (from working 500K dataset) ===

HIGH_RISK_FRAUD = ["mule_account", "card_fraud", "identity_theft", "account_takeover", "synthetic_identity"]
LOW_RISK_FRAUD = ["suspicious_transfer", "unusual_activity"]

HIGH_FRAUD_WEIGHTS = {
    "mule_account": 0.25,
    "card_fraud": 0.24,
    "identity_theft": 0.15,
    "account_takeover": 0.19,
    "synthetic_identity": 0.17,
}

LOW_FRAUD_WEIGHTS = {
    "suspicious_transfer": 0.51,
    "unusual_activity": 0.49,
}

JURISDICTIONS = {
    "JP": 0.118,
    "AU": 0.118,
    "DE": 0.117,
    "FR": 0.116,
    "US": 0.116,
    "GB": 0.114,
    "BR": 0.083,
    "ES": 0.078,
    "HK": 0.036,
    "CH": 0.035,
    "SG": 0.035,
    "CA": 0.034,
}

REPORTING_JUR_NONE_RATE = 0.15
BANKS = [f"BANK{i:03d}" for i in range(1, 51)]
YEARS = {
    2018: 0.133,
    2019: 0.093,
    2020: 0.143,
    2021: 0.142,
    2022: 0.139,
    2023: 0.144,
    2024: 0.147,
    2025: 0.059,
}
MONTHS = list(range(1, 13))
DAYS = list(range(1, 29))
IS_ACTIVE_RATE = 0.503

INT_COLS = {"risk_level", "year", "month", "day"}
ALL_COLS = [
    "report_id",
    "reporting_bank_id",
    "reporting_jurisdiction",
    "reported_iban",
    "account_jurisdiction",
    "risk_level",
    "fraud_type",
    "year",
    "month",
    "day",
    "is_active",
]


def weighted_choice(rng, weights_dict):
    items = list(weights_dict.keys())
    weights = list(weights_dict.values())
    return rng.choices(items, weights=weights, k=1)[0]


def generate_record(rng, report_id):
    is_high_risk = rng.random() < 0.6534

    if is_high_risk:
        risk_level = rng.randint(50, 100)
        fraud_type = weighted_choice(rng, HIGH_FRAUD_WEIGHTS)
    else:
        risk_level = rng.randint(0, 49)
        fraud_type = weighted_choice(rng, LOW_FRAUD_WEIGHTS)

    account_jur = weighted_choice(rng, JURISDICTIONS)
    reporting_jur = "" if rng.random() < REPORTING_JUR_NONE_RATE else weighted_choice(rng, JURISDICTIONS)

    return {
        "report_id": str(report_id),
        "reporting_bank_id": rng.choice(BANKS),
        "reporting_jurisdiction": reporting_jur,
        "reported_iban": f"IBAN{rng.randint(10000000, 99999999)}",
        "account_jurisdiction": account_jur,
        "risk_level": risk_level,
        "fraud_type": fraud_type,
        "year": weighted_choice(rng, YEARS),
        "month": rng.choice(MONTHS),
        "day": rng.choice(DAYS),
        "is_active": "true" if rng.random() < IS_ACTIVE_RATE else "false",
    }


def generate_batch(batch_seed, count, start_id):
    """Generate a batch of records with its own RNG for distribution consistency."""
    rng = random.Random(batch_seed)
    return [generate_record(rng, start_id + i) for i in range(count)]


def generate_noisy_record(rng, report_id):
    """Generate a record with fraud_type drawn from the OPPOSITE risk class."""
    is_high_risk = rng.random() < 0.6534

    if is_high_risk:
        risk_level = rng.randint(50, 100)
        fraud_type = weighted_choice(rng, LOW_FRAUD_WEIGHTS)
    else:
        risk_level = rng.randint(0, 49)
        fraud_type = weighted_choice(rng, HIGH_FRAUD_WEIGHTS)

    account_jur = weighted_choice(rng, JURISDICTIONS)
    reporting_jur = "" if rng.random() < REPORTING_JUR_NONE_RATE else weighted_choice(rng, JURISDICTIONS)

    return {
        "report_id": str(report_id),
        "reporting_bank_id": rng.choice(BANKS),
        "reporting_jurisdiction": reporting_jur,
        "reported_iban": f"IBAN{rng.randint(10000000, 99999999)}",
        "account_jurisdiction": account_jur,
        "risk_level": risk_level,
        "fraud_type": fraud_type,
        "year": weighted_choice(rng, YEARS),
        "month": rng.choice(MONTHS),
        "day": rng.choice(DAYS),
        "is_active": "true" if rng.random() < IS_ACTIVE_RATE else "false",
    }


def append_to_sqlite(records, db_path, table_name):
    """Append records to an existing SQLite table."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    placeholders = ", ".join("?" for _ in ALL_COLS)
    insert_sql = f"INSERT INTO {table_name} ({', '.join(ALL_COLS)}) VALUES ({placeholders})"

    batch_size = 10_000
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        rows = [tuple(int(rec[c]) if c in INT_COLS else str(rec[c]) for c in ALL_COLS) for rec in batch]
        cur.executemany(insert_sql, rows)
        conn.commit()
    conn.close()


def append_noise():
    """Append noisy records (flipped fraud_type) to existing DBs and write JSON batches."""
    sys.stdout.reconfigure(line_buffering=True)

    conn = sqlite3.connect(str(TRAIN_DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT MAX(CAST(report_id AS INTEGER)) FROM train")
    train_max = cur.fetchone()[0]
    conn.close()

    conn = sqlite3.connect(str(TEST_DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT MAX(CAST(report_id AS INTEGER)) FROM test")
    test_max = cur.fetchone()[0]
    conn.close()

    global_max = max(train_max, test_max)

    print(f"Current max report_id: {global_max}")
    print(f"Generating {NOISE_TRAIN:,} noisy training records...")
    rng = random.Random(NOISE_SEED)
    train_noise = [generate_noisy_record(rng, global_max + 1 + i) for i in range(NOISE_TRAIN)]

    print(f"Generating {NOISE_TEST:,} noisy test records...")
    test_noise = [generate_noisy_record(rng, global_max + NOISE_TRAIN + 1 + i) for i in range(NOISE_TEST)]

    print(f"Appending to {TRAIN_DB_PATH}...")
    append_to_sqlite(train_noise, TRAIN_DB_PATH, "train")

    print(f"Appending to {TEST_DB_PATH}...")
    append_to_sqlite(test_noise, TEST_DB_PATH, "test")

    JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    noise_batch_size = 50_000
    for i in range(0, len(train_noise), noise_batch_size):
        batch = train_noise[i : i + noise_batch_size]
        idx = i // noise_batch_size + 1
        fp = JSON_OUTPUT_DIR / f"fraud_train_noise_{idx:02d}.json"
        size = write_json_batch(batch, fp, SCHEMA_PLACEHOLDER)
        print(f"  {fp.name}: {len(batch):,} records ({size / 1024 / 1024:.1f} MB)")

    fp = JSON_OUTPUT_DIR / "fraud_test_noise_01.json"
    size = write_json_batch(test_noise, fp, SCHEMA_PLACEHOLDER)
    print(f"  {fp.name}: {len(test_noise):,} records ({size / 1024 / 1024:.1f} MB)")

    for db_path, table in [(TRAIN_DB_PATH, "train"), (TEST_DB_PATH, "test")]:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        total = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE risk_level >= 50")
        high = cur.fetchone()[0]
        conn.close()
        print(f"  {db_path}: {total:,} records, {high / total * 100:.1f}% high risk")

    print("Done.")


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

    batch_size = 10_000
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        rows = []
        for rec in batch:
            row = tuple(int(rec[c]) if c in INT_COLS else str(rec[c]) for c in ALL_COLS)
            rows.append(row)
        cur.executemany(insert_sql, rows)
        conn.commit()

    conn.close()


def verify_batch_distributions(all_batches):
    """Print per-batch high-risk % to confirm distribution consistency."""
    print("\nPer-batch distribution verification:")
    print(f"  {'Batch':<12} {'Records':>8} {'High%':>7} {'Low%':>7}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 7} {'-' * 7}")
    for label, records in all_batches:
        high = sum(1 for r in records if r["risk_level"] >= 50)
        low = len(records) - high
        pct_high = high / len(records) * 100
        pct_low = low / len(records) * 100
        print(f"  {label:<12} {len(records):>8,} {pct_high:>6.1f}% {pct_low:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate demo data for Blind Insight ML notebook")
    parser.add_argument("--json-only", action="store_true", help="Only generate JSON batches")
    parser.add_argument("--sqlite-only", action="store_true", help="Only generate SQLite databases")
    parser.add_argument(
        "--append-noise", action="store_true", help="Append noisy records (flipped fraud_type) to existing DBs"
    )
    args = parser.parse_args()

    if args.append_noise:
        append_noise()
        return

    sys.stdout.reconfigure(line_buffering=True)
    total_train = TRAIN_BATCHES * TRAIN_BATCH_SIZE

    print(f"Generating {total_train:,} training records ({TRAIN_BATCHES} batches x {TRAIN_BATCH_SIZE:,})...")
    print(f"Generating {TEST_RECORDS:,} test records (1 batch)...")
    print("Each batch seeded independently for consistent distributions.\n")

    # --- Generate training batches ---
    train_batches = []
    all_train_records = []
    for batch_idx in range(TRAIN_BATCHES):
        batch_seed = BASE_SEED + batch_idx
        start_id = batch_idx * TRAIN_BATCH_SIZE + 1
        records = generate_batch(batch_seed, TRAIN_BATCH_SIZE, start_id)
        train_batches.append((f"train_{batch_idx + 1:02d}", records))
        all_train_records.extend(records)

    # --- Generate test batch ---
    test_seed = BASE_SEED + 100
    test_start_id = total_train + 1
    test_records = generate_batch(test_seed, TEST_RECORDS, test_start_id)

    # --- Verify distributions ---
    verify_batch_distributions(train_batches + [("test", test_records)])

    # --- Write JSON batches ---
    if not args.sqlite_only:
        JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting JSON batches to {JSON_OUTPUT_DIR}/...")

        for batch_idx, (label, records) in enumerate(train_batches, 1):
            filepath = JSON_OUTPUT_DIR / f"fraud_train_batch_{batch_idx:02d}.json"
            size = write_json_batch(records, filepath, SCHEMA_PLACEHOLDER)
            size_mb = size / (1024 * 1024)
            print(f"  {filepath.name}: {len(records):,} records ({size_mb:.1f} MB)")

        test_path = JSON_OUTPUT_DIR / "fraud_test_batch_01.json"
        size = write_json_batch(test_records, test_path, SCHEMA_PLACEHOLDER)
        size_mb = size / (1024 * 1024)
        print(f"  {test_path.name}: {len(test_records):,} records ({size_mb:.1f} MB)")

    # --- Write SQLite databases ---
    if not args.json_only:
        print(f"\nBuilding {TRAIN_DB_PATH}...")
        write_sqlite(all_train_records, TRAIN_DB_PATH, "train")
        print(f"  Done: {total_train:,} records")

        print(f"Building {TEST_DB_PATH}...")
        write_sqlite(test_records, TEST_DB_PATH, "test")
        print(f"  Done: {TEST_RECORDS:,} records")

    # --- Final verification ---
    if not args.json_only:
        print("\nSQLite verification:")
        for db_path, table, expected in [
            (TRAIN_DB_PATH, "train", total_train),
            (TEST_DB_PATH, "test", TEST_RECORDS),
        ]:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE risk_level >= 50")
            high = cur.fetchone()[0]
            conn.close()
            pct = high / count * 100
            status = "OK" if count == expected else "MISMATCH"
            print(f"  {db_path}: {count:,} records, {pct:.1f}% high risk [{status}]")

        # Verify batch-order alignment: first 50K in SQLite should match batch 1
        conn = sqlite3.connect(str(TRAIN_DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM train WHERE rowid <= 50000 AND risk_level >= 50")
        first_50k_high = cur.fetchone()[0]
        conn.close()
        batch1_high = sum(1 for r in train_batches[0][1] if r["risk_level"] >= 50)
        match = "OK" if first_50k_high == batch1_high else "MISMATCH"
        print(f"  Batch-order check: SQLite rows 1-50K high={first_50k_high}, batch_01 high={batch1_high} [{match}]")


if __name__ == "__main__":
    main()
