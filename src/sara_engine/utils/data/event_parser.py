# Directory Path: src/sara_engine/utils/data/event_parser.py
# English Title: DVS Event Data Parser
# Purpose/Content: Tools for parsing event-based data (DVS) from various formats 
# (AEDAT, CSV) into SNN-compatible spike trains.

import csv
from typing import List, Tuple, Dict, Any

class DVSEventParser:
    """
    Parses event-based data (x, y, t, p) into structured formats for SNN processing.
    """

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
