#!/usr/bin/env python3
"""Execute the frozen source/support audit without exposing sealed values."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.appliances_prospective import extract
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, raw_data_path

PROTOCOL_SHA256 = "2ccb958df09bc6de92744e6a5fbc7c23454b428b2bee951f7cf4436411418caa"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify the retained receipt without rewriting it")
    args = parser.parse_args()
    protocol_path = Path(processed_data_path("benchmark_fixtures", "appliances_prospective_v1.json"))
    output = Path(ensure_parent_directory(processed_data_path("benchmark_fixtures", "appliances_prospective_receipt_v1.json")))
    if output.exists() and not args.verify:
        raise ValueError("Retained receipt exists; use --verify")
    protocol_bytes = protocol_path.read_bytes()
    if sha256(protocol_bytes).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol checksum mismatch")
    protocol = json.loads(protocol_bytes)
    source = Path(raw_data_path("r2_appliances", "energydata_complete.csv"))
    if sha256(source.read_bytes()).hexdigest() != protocol["source_sha256"]:
        raise ValueError("Source checksum mismatch")
    with source.open(newline="", encoding="utf-8-sig") as stream:
        examples, result = extract(csv.reader(stream), protocol)
    # Hash the causal examples without retaining raw values in the receipt.
    digest = sha256()
    for example in examples:
        digest.update(repr(example).encode())
    result.update(protocol_sha256=sha256(protocol_bytes).hexdigest(),
                  source_sha256=protocol["source_sha256"], examples_sha256=digest.hexdigest())
    if args.verify:
        if json.loads(output.read_text()) != result:
            raise ValueError("Retained receipt does not match causal replay")
    else:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
