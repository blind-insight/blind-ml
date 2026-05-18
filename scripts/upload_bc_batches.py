"""
Serial upload script — breast cancer training batches.
5 batches x 20K records = 100K total.
Polls each job to "complete" before waiting 2 minutes and moving to the next.
"""

import fcntl
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from repo root — no dotenv dependency required
env_path = REPO_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

PROXY_URL = os.getenv("BI_PROXY_URL", "https://local.blindinsight.io")
EMAIL = os.getenv("BI_EMAIL")
PASSWORD = os.getenv("BI_PASSWORD")
BATCH_DELAY = 120  # 2 minutes between batches (conservative — shaky server)
POLL_EVERY = 10  # seconds between status polls
LOCK_FILE = "/tmp/bi_bc_upload.lock"

BATCH_DIR = REPO_ROOT / "demo_data" / "upload_batches"
BATCHES = sorted(BATCH_DIR.glob("bc_train_batch_*.json"))

# ── lockfile guard ────────────────────────────────────────────────────────────
lock_fd = open(LOCK_FILE, "w")
try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("ERROR: Another upload process is already running. Aborting.")
    sys.exit(1)

if not EMAIL or not PASSWORD:
    print("ERROR: Set BI_EMAIL and BI_PASSWORD in .env (repo root).")
    sys.exit(1)

if not BATCHES:
    print(f"ERROR: No bc_train_batch_*.json files in {BATCH_DIR}")
    print("Run: python3 scripts/generate_healthcare_data.py")
    sys.exit(1)

auth = (EMAIL, PASSWORD)


def poll_job(job_id):
    start = time.time()
    while True:
        time.sleep(POLL_EVERY)
        resp = requests.get(f"{PROXY_URL}/api/jobs/{job_id}/", auth=auth, verify=False)
        resp.raise_for_status()
        job = resp.json()
        elapsed = int(time.time() - start)
        print(f"    [{elapsed:>3}s] {job.get('processed', '?')}/{job.get('total', '?')} — {job['status']}")
        if job["status"] == "complete":
            return True
        if job["status"] == "failed":
            print(f"  JOB FAILED: {job}")
            return False


print(f"Uploading {len(BATCHES)} training batches to {PROXY_URL}")
print(f"Batch delay: {BATCH_DELAY}s | Poll interval: {POLL_EVERY}s\n")

for i, batch_path in enumerate(BATCHES):
    print(f"[{i + 1}/{len(BATCHES)}] Loading {batch_path.name}...")
    with open(batch_path) as f:
        records = json.load(f)
    print(f"  {len(records):,} records — posting to {PROXY_URL}/api/jobs/upload/")

    resp = requests.post(
        f"{PROXY_URL}/api/jobs/upload/",
        json=records,
        auth=auth,
        verify=False,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"  Job ID: {job_id} — polling...")

    ok = poll_job(job_id)
    if not ok:
        print(f"\nAborting after batch {i + 1} failure.")
        sys.exit(1)

    print(f"  Batch {i + 1} complete.")

    if i < len(BATCHES) - 1:
        print(f"  Waiting {BATCH_DELAY}s before next batch...\n")
        time.sleep(BATCH_DELAY)

print(f"\nAll {len(BATCHES)} batches uploaded successfully.")
