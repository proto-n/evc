import torch

class TimestampedAudioBuffer:
    """
    A simple buffer for timestamped audio chunks.
    It assumes that chunks arrive in order.
    When retrieving a segment, if there are missing pieces,
    the corresponding samples are left as zeros.
    """
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.chunks = []  # list of (timestamp, audio tensor)
        self.current_time = None  # left edge of the buffered timeline

    def add_chunk(self, timestamp, chunk):
        # If first chunk, initialize our starting time.
        if self.current_time is None:
            self.current_time = timestamp
        self.chunks.append((timestamp, chunk))

    def get_segment(self, duration):
        """
        Returns a tensor covering 'duration' seconds starting from current_time.
        Missing parts are filled with zeros.
        """
        total_samples = int(duration * self.sample_rate)
        segment = torch.zeros(total_samples)
        if self.current_time is None:
            return segment

        desired_start = self.current_time
        desired_end = desired_start + duration

        for ts, chunk in self.chunks:
            chunk_duration = len(chunk) / self.sample_rate
            chunk_end = ts + chunk_duration

            # Determine the overlapping time between the chunk and the desired segment.
            inter_start = max(desired_start, ts)
            inter_end = min(desired_end, chunk_end)

            if inter_start < inter_end:
                out_start = int((inter_start - desired_start) * self.sample_rate)
                out_end = int((inter_end - desired_start) * self.sample_rate)
                in_start = int((inter_start - ts) * self.sample_rate)
                in_end = in_start + (out_end - out_start)
                segment[out_start:out_end] = chunk[in_start:in_end]

        return segment

    def pop_segment(self, duration):
        """
        Removes data corresponding to 'duration' seconds from the left of the buffer.
        Returns the same segment (with zeros in any missing areas) that get_segment would.
        """
        segment = self.get_segment(duration)
        new_current = self.current_time + duration if self.current_time is not None else duration
        new_chunks = []
        for ts, chunk in self.chunks:
            chunk_duration = len(chunk) / self.sample_rate
            chunk_end = ts + chunk_duration
            if chunk_end <= new_current:
                # Entire chunk is before the new pointer – drop it.
                continue
            elif ts < new_current:
                # Partially consumed chunk; drop the consumed part.
                consumed_samples = int((new_current - ts) * self.sample_rate)
                new_chunk = chunk[consumed_samples:]
                new_chunks.append((new_current, new_chunk))
            else:
                # Chunk starts entirely after new_current; keep it as is.
                new_chunks.append((ts, chunk))
        self.chunks = new_chunks
        self.current_time = new_current
        return segment