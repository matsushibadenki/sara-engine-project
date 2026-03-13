# Directory Path: src/sara_engine/utils/data/event_transform.py
# English Title: Event Data Transformation Tools
# Purpose/Content: Spatial and temporal transformation tools for event-based data, 
# including downsampling and resolution reduction.

from typing import List, Tuple

class EventDownsampler:
    """
    Provides spatial and temporal downsampling for event streams.
    """

    def spatial_downsample(
        self, 
        events: List[Tuple[float, int, int, int]], 
        factor: int = 2
    ) -> List[Tuple[float, int, int, int]]:
        """
        Reduces spatial resolution by dividing x and y coordinates.
        """
        if factor <= 1:
            return events
            
        downsampled: List[Tuple[float, int, int, int]] = []
        for t, x, y, p in events:
            downsampled.append((t, x // factor, y // factor, p))
        return downsampled

    def temporal_denoise(
        self, 
        events: List[Tuple[float, int, int, int]], 
        dt_threshold: float = 0.01
    ) -> List[Tuple[float, int, int, int]]:
        """
        Removes events that don't have enough neighbors in space-time (Noise reduction).
        Simplification: Filters events with the same coordinate if they arrive too fast.
        """
        last_times: dict[tuple[int, int], float] = {}
        filtered: List[Tuple[float, int, int, int]] = []
        
        for t, x, y, p in events:
            coord = (x, y)
            if coord in last_times:
                if t - last_times[coord] < dt_threshold:
                    continue
            last_times[coord] = t
            filtered.append((t, x, y, p))
            
        return filtered

    def temporal_downsample(
        self,
        events: List[Tuple[float, int, int, int]],
        time_step: float = 0.01,
    ) -> List[Tuple[float, int, int, int]]:
        """
        Bins timestamps into discrete steps to reduce temporal resolution.
        """
        if time_step <= 0:
            return events

        downsampled: List[Tuple[float, int, int, int]] = []
        for t, x, y, p in events:
            binned_t = (t // time_step) * time_step
            downsampled.append((binned_t, x, y, p))
        return downsampled

    def crop(
        self, 
        events: List[Tuple[float, int, int, int]], 
        x_range: Tuple[int, int], 
        y_range: Tuple[int, int]
    ) -> List[Tuple[float, int, int, int]]:
        """Crops events within a specific spatial box."""
        cropped = []
        for t, x, y, p in events:
            if x_range[0] <= x < x_range[1] and y_range[0] <= y < y_range[1]:
                # Offset to new origin
                cropped.append((t, x - x_range[0], y - y_range[0], p))
        return cropped
