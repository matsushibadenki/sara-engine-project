#!/usr/bin/env python3
"""Acquire only the pinned official public source; no feature inspection."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, raw_data_path


def main() -> None:
    protocol_bytes = Path(processed_data_path("benchmark_fixtures", "appliances_prospective_v1.json")).read_bytes()
    if sha256(protocol_bytes).hexdigest() != "2ccb958df09bc6de92744e6a5fbc7c23454b428b2bee951f7cf4436411418caa":
        raise ValueError("Frozen protocol checksum mismatch")
    protocol = json.loads(protocol_bytes)
    target = Path(ensure_parent_directory(raw_data_path("r2_appliances", "energydata_complete.csv")))
    with urlopen(protocol["source_url"], timeout=60) as response:
        archive = response.read(20_000_001)
    if len(archive) > 20_000_000:
        raise ValueError("Source archive exceeds bounded download limit")
    with ZipFile(BytesIO(archive)) as zipped:
        member = zipped.getinfo("energydata_complete.csv")
        if member.file_size > 20_000_000:
            raise ValueError("Source CSV exceeds bounded extraction limit")
        payload = zipped.read(member)
    if sha256(payload).hexdigest() != protocol["source_sha256"]:
        raise ValueError("Official source checksum mismatch")
    with target.open("xb") as stream:
        stream.write(payload)
    print("Pinned appliance source acquired; no feature values inspected")


if __name__ == "__main__":
    main()
