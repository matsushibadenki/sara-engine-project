# Directory Path: src/sara_engine/utils/data/event_parser.py
# English Title: DVS Event Data Parser
# Purpose/Content: Tools for parsing event-based data (DVS) from various formats 
# (AEDAT, CSV) into SNN-compatible spike trains.

import csv
import json
import struct
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

np: Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled for minimal environments
    np = None


@dataclass
class AERAddressConfig:
    """
    Defines how to decode an AER address into x, y, and polarity values.
    """

    x_mask: int
    x_shift: int
    y_mask: int
    y_shift: int
    p_mask: int
    p_shift: int

    def decode(self, address: int) -> Tuple[int, int, int]:
        x = (address & self.x_mask) >> self.x_shift
        y = (address & self.y_mask) >> self.y_shift
        polarity = (address & self.p_mask) >> self.p_shift
        return int(x), int(y), int(polarity)

class DVSEventParser:
    """
    Parses event-based data (x, y, t, p) into structured formats for SNN processing.
    """

    DEFAULT_DVS128_CONFIG = AERAddressConfig(
        x_mask=0x7F,
        x_shift=0,
        y_mask=0x3F80,
        y_shift=7,
        p_mask=0x4000,
        p_shift=14,
    )

    def parse_csv_events(self, file_path: str) -> List[Tuple[float, int, int, int]]:
        """
        Parses events from a CSV file. Expected header: t, x, y, p.
        Returns a list of (timestamp, x, y, polarity).
        """
        events: List[Tuple[float, int, int, int]] = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = float(row['t'])
                    x = int(row['x'])
                    y = int(row['y'])
                    p = int(row['p'])
                    events.append((t, x, y, p))
        except Exception as e:
            print(f"[DVSEventParser] Error parsing CSV: {e}")
        return events

    def parse_npz_events(self, file_path: str) -> List[Tuple[float, int, int, int]]:
        """
        Parses events from an NPZ file.
        Expected keys: t, x, y, p or a single 'events' array with shape (N, 4).
        """
        events: List[Tuple[float, int, int, int]] = []
        if np is None:
            print("[DVSEventParser] NumPy is required to parse NPZ files.")
            return events
        try:
            data = np.load(file_path, allow_pickle=True)
            if "events" in data:
                arr = np.asarray(data["events"])
                if arr.ndim != 2 or arr.shape[1] != 4:
                    raise ValueError("events array must be shaped (N, 4)")
                for row in arr:
                    t, x, y, p = row
                    events.append((float(t), int(x), int(y), int(p)))
            else:
                required = ("t", "x", "y", "p")
                if not all(key in data for key in required):
                    raise ValueError("NPZ must contain keys: t, x, y, p or events")
                ts = data["t"]
                xs = data["x"]
                ys = data["y"]
                ps = data["p"]
                for t, x, y, p in zip(ts, xs, ys, ps):
                    events.append((float(t), int(x), int(y), int(p)))
        except Exception as e:
            print(f"[DVSEventParser] Error parsing NPZ: {e}")
        return events

    def parse_aedat_events(
        self,
        file_path: str,
        config: Optional[AERAddressConfig] = None,
        endian: str = ">",
        timestamp_scale: float = 1.0,
        max_events: Optional[int] = None,
    ) -> List[Tuple[float, int, int, int]]:
        """
        Parses events from an AEDAT/AER file using a configurable address layout.
        """
        events: List[Tuple[float, int, int, int]] = []
        layout = config or self.DEFAULT_DVS128_CONFIG
        unpacker = struct.Struct(f"{endian}II")

        try:
            with open(file_path, "rb") as f:
                pos = f.tell()
                line = f.readline()
                while line.startswith(b"#"):
                    pos = f.tell()
                    line = f.readline()
                if line:
                    f.seek(pos)

                data = f.read()
        except Exception as e:
            print(f"[DVSEventParser] Error reading AEDAT: {e}")
            return events

        count = 0
        for offset in range(0, len(data) - 7, 8):
            addr, ts = unpacker.unpack_from(data, offset)
            x, y, polarity = layout.decode(addr)
            events.append((float(ts) * timestamp_scale, x, y, polarity))
            count += 1
            if max_events and count >= max_events:
                break

        return events

    def to_spike_trains(
        self, 
        events: List[Tuple[float, int, int, int]], 
        width: int, 
        height: int, 
        time_step: float = 1.0
    ) -> Dict[float, List[int]]:
        """
        Converts raw events into temporal bins of spike IDs.
        Spike ID is calculated as: y * width + x (+ width * height if polarity is negative).
        """
        spike_trains: Dict[float, List[int]] = {}
        for t, x, y, p in events:
            # Quantize time
            bin_t = (t // time_step) * time_step
            if bin_t not in spike_trains:
                spike_trains[bin_t] = []
            
            # Spatial ID
            spike_id = y * width + x
            if p == 0: # Assuming 0/1 for polarity
                spike_id += (width * height) # Offset for negative polarity
                
            spike_trains[bin_t].append(spike_id)
            
        return spike_trains

    def get_metadata(self, events: List[Tuple[float, int, int, int]]) -> Dict[str, Any]:
        """Calculates basic metadata from events."""
        if not events:
            return {}
        ts = [e[0] for e in events]
        xs = [e[1] for e in events]
        ys = [e[2] for e in events]
        return {
            "count": len(events),
            "duration": max(ts) - min(ts),
            "width": max(xs) + 1,
            "height": max(ys) + 1,
            "min_t": min(ts),
            "max_t": max(ts)
        }

    def load_layout_from_json(self, json_path: str) -> Optional[AERAddressConfig]:
        """
        Loads an AER address layout from a JSON file.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AERAddressConfig(
                x_mask=int(data["x_mask"]),
                x_shift=int(data["x_shift"]),
                y_mask=int(data["y_mask"]),
                y_shift=int(data["y_shift"]),
                p_mask=int(data["p_mask"]),
                p_shift=int(data["p_shift"]),
            )
        except Exception as e:
            print(f"[DVSEventParser] Error loading layout JSON: {e}")
            return None
