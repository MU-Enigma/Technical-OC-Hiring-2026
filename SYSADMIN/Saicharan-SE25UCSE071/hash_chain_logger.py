#!/usr/bin/env python3
# hash_chain_logger.py
#
# Wraps /var/log/assetvault.log in a SHA-256 hash chain, so the incident
# record can be independently verified as unaltered after the fact. Each
# entry's hash incorporates the previous entry's hash -- any retroactive
# edit to the chain file breaks verification from that point forward.
#
# Run continuously as a background process:
#   sudo -u svc_vault nohup python3 hash_chain_logger.py > /dev/null 2>&1 &
#
# Verify integrity at any time:
#   python3 hash_chain_logger.py verify

import hashlib
import json
import time
import subprocess
import sys

CHAIN_FILE = "/var/log/assetvault_chain.jsonl"
SOURCE_LOG = "/var/log/assetvault.log"
GENESIS = "0" * 64


def last_hash():
    try:
        with open(CHAIN_FILE, "rb") as f:
            last = f.readlines()[-1]
            return json.loads(last)["hash"]
    except (FileNotFoundError, IndexError):
        return GENESIS


def append_entry(raw_line):
    prev = last_hash()
    digest = hashlib.sha256((prev + raw_line).encode()).hexdigest()
    entry = {
        "ts": time.time(),
        "line": raw_line,
        "prev_hash": prev,
        "hash": digest,
    }
    with open(CHAIN_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def verify_chain():
    prev = GENESIS
    try:
        with open(CHAIN_FILE) as f:
            i = -1
            for i, row in enumerate(f):
                entry = json.loads(row)
                expected = hashlib.sha256((prev + entry["line"]).encode()).hexdigest()
                if expected != entry["hash"] or entry["prev_hash"] != prev:
                    print(f"CHAIN BROKEN at entry {i}")
                    return False
                prev = entry["hash"]
            print(f"Chain OK -- {i + 1} entries verified, no tampering detected.")
            return True
    except FileNotFoundError:
        print(f"No chain file found at {CHAIN_FILE}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_chain()
    else:
        # tail -F survives the source log being rotated/truncated and
        # re-opens automatically -- important since assetvault.log is
        # sometimes cleared between demo runs
        proc = subprocess.Popen(
            ["tail", "-F", SOURCE_LOG], stdout=subprocess.PIPE, text=True
        )
        for line in proc.stdout:
            append_entry(line.strip())
