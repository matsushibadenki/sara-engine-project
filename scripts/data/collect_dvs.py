# Directory Path: scripts/data/collect_dvs.py
# English Title: DVS Dataset Collector
# Purpose/Content: Parses event-based datasets (CSV/NPZ/AEDAT) into spike trains and
# stores them under data/processed for downstream SNN training.

import argparse
import json
import os
import sys
from typing import Iterable, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "src"
)))

from sara_engine.utils.project_paths import (  # noqa: E402
    raw_data_path,
    processed_data_path,
    ensure_output_directory,
    ensure_parent_directory,
)
from sara_engine.utils.data.event_parser import DVSEventParser, AERAddressConfig  # noqa: E402
from sara_engine.utils.data.event_transform import EventDownsampler  # noqa: E402


SUPPORTED_EXTS = (".csv", ".npz", ".aedat", ".aer", ".dat")


def iter_event_files(input_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(SUPPORTED_EXTS):
                yield os.path.join(root, name)


def parse_crop(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError("crop must be in the form x0,x1,y0,y1")
    return tuple(int(p.strip()) for p in parts)  # type: ignore[return-value]


def load_aer_config(
    layout: str,
    config_path: Optional[str],
    parser: DVSEventParser,
) -> Optional[AERAddressConfig]:
    if layout == "dvs128":
        return parser.DEFAULT_DVS128_CONFIG
    if layout == "custom":
        if not config_path:
            raise ValueError("custom layout requires --aedat-config path")
        return parser.load_layout_from_json(config_path)
    if layout == "none":
        return None
    raise ValueError(f"Unknown AEDAT layout: {layout}")


def write_spike_trains_jsonl(
    output_path: str,
    spike_trains: dict[float, List[int]],
) -> None:
    ensure_parent_directory(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in sorted(spike_trains.keys()):
            payload = {"t": float(t), "spikes": spike_trains[t]}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect DVS event data and convert to spike trains."
    )
    parser.add_argument(
        "--input-dir",
        default=raw_data_path("dvs"),
        help="Directory containing raw DVS files.",
    )
    parser.add_argument(
        "--output-dir",
        default=processed_data_path("dvs"),
        help="Directory to store processed spike trains.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.01,
        help="Temporal bin size for spike train generation.",
    )
    parser.add_argument(
        "--spatial-downsample",
        type=int,
        default=1,
        help="Spatial downsampling factor.",
    )
    parser.add_argument(
        "--temporal-denoise",
        type=float,
        default=0.0,
        help="Minimum time between events at the same pixel.",
    )
    parser.add_argument(
        "--temporal-downsample",
        type=float,
        default=0.0,
        help="Temporal bin size for downsampling before spike conversion.",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Crop box in the form x0,x1,y0,y1.",
    )
    parser.add_argument(
        "--aedat-layout",
        default="dvs128",
        choices=("dvs128", "custom", "none"),
        help="Address layout for AEDAT/AER files.",
    )
    parser.add_argument(
        "--aedat-config",
        default=None,
        help="Path to a JSON file describing the AEDAT address layout.",
    )
    parser.add_argument(
        "--aedat-endian",
        default=">",
        choices=(">", "<"),
        help="Endian for AEDAT parsing (default: big endian).",
    )
    parser.add_argument(
        "--timestamp-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to AEDAT timestamps.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional maximum number of events per file.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    ensure_output_directory(output_dir)

    dvs_parser = DVSEventParser()
    downsampler = EventDownsampler()
    crop_box = parse_crop(args.crop)
    aer_config = load_aer_config(args.aedat_layout, args.aedat_config, dvs_parser)

    files = list(iter_event_files(input_dir))
    if not files:
        print(f"[collect_dvs] No event files found in {input_dir}")
        return

    index_path = processed_data_path("dvs", "index.jsonl")
    ensure_parent_directory(index_path)

    processed_count = 0
    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            events = dvs_parser.parse_csv_events(file_path)
        elif ext == ".npz":
            events = dvs_parser.parse_npz_events(file_path)
        else:
            events = dvs_parser.parse_aedat_events(
                file_path,
                config=aer_config,
                endian=args.aedat_endian,
                timestamp_scale=args.timestamp_scale,
                max_events=args.max_events,
            )

        if args.max_events and len(events) > args.max_events:
            events = events[:args.max_events]

        if not events:
            print(f"[collect_dvs] Skipping empty file: {file_path}")
            continue

        if args.spatial_downsample > 1:
            events = downsampler.spatial_downsample(
                events, factor=args.spatial_downsample
            )
        if args.temporal_denoise > 0:
            events = downsampler.temporal_denoise(
                events, dt_threshold=args.temporal_denoise
            )
        if args.temporal_downsample > 0:
            events = downsampler.temporal_downsample(
                events, time_step=args.temporal_downsample
            )
        if crop_box:
            x0, x1, y0, y1 = crop_box
            events = downsampler.crop(events, (x0, x1), (y0, y1))

        metadata = dvs_parser.get_metadata(events)
        if not metadata:
            print(f"[collect_dvs] No metadata available for: {file_path}")
            continue

        width = int(metadata["width"])
        height = int(metadata["height"])
        if width <= 0 or height <= 0:
            print(f"[collect_dvs] Invalid width/height in {file_path}")
            continue

        spike_trains = dvs_parser.to_spike_trains(
            events,
            width=width,
            height=height,
            time_step=args.time_step,
        )

        stem = os.path.splitext(os.path.basename(file_path))[0]
        spike_path = processed_data_path("dvs", "spike_trains", f"{stem}.jsonl")
        meta_path = processed_data_path("dvs", "metadata", f"{stem}.json")

        write_spike_trains_jsonl(spike_path, spike_trains)
        ensure_parent_directory(meta_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_path": file_path,
                    "event_count": len(events),
                    "width": width,
                    "height": height,
                    "time_step": args.time_step,
                    "spike_bins": len(spike_trains),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(index_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "source_path": file_path,
                        "spike_train_path": spike_path,
                        "metadata_path": meta_path,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        processed_count += 1
        print(f"[collect_dvs] Processed {file_path} -> {spike_path}")

    print(f"[collect_dvs] Completed. Files processed: {processed_count}")


if __name__ == "__main__":
    main()
