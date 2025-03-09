import pytest
import torch
from evc.timestamped_audio_buffer import TimestampedAudioBuffer

@pytest.fixture
def buffer():
    return TimestampedAudioBuffer(sample_rate=22050)

def test_initial_state(buffer):
    assert buffer.current_time is None
    assert len(buffer.chunks) == 0

def test_add_first_chunk(buffer):
    chunk = torch.ones(22050)  # 1 second of audio
    timestamp = 1000.0
    buffer.add_chunk(timestamp, chunk)
    
    assert buffer.current_time == timestamp
    assert len(buffer.chunks) == 1
    assert buffer.chunks[0][0] == timestamp
    assert torch.equal(buffer.chunks[0][1], chunk)

def test_get_segment_empty_buffer(buffer):
    segment = buffer.get_segment(1.0)
    assert len(segment) == 22050
    assert torch.all(segment == 0)

def test_get_segment_complete(buffer):
    # Add 1 second of ones
    chunk = torch.ones(22050)
    buffer.add_chunk(1000.0, chunk)
    
    segment = buffer.get_segment(1.0)
    assert len(segment) == 22050
    assert torch.all(segment == 1)

def test_pop_segment(buffer):
    # Add 2 seconds of audio
    chunk1 = torch.ones(22050)
    chunk2 = torch.ones(22050) * 2
    buffer.add_chunk(1000.0, chunk1)
    buffer.add_chunk(1001.0, chunk2)
    
    # Pop 1 second
    segment = buffer.pop_segment(1.0)
    assert len(segment) == 22050
    assert torch.all(segment == 1)
    assert len(buffer.chunks) == 1
    assert buffer.current_time == 1001.0

def test_partial_segment(buffer):
    # Add 0.5 seconds of audio
    chunk = torch.ones(11025)  # half second
    buffer.add_chunk(1000.0, chunk)
    
    # Request 1 second
    segment = buffer.get_segment(1.0)
    assert len(segment) == 22050
    assert torch.all(segment[:11025] == 1)  # First half ones
    assert torch.all(segment[11025:] == 0)  # Second half zeros

def test_chunk_splitting(buffer):
    # Add 1.5 seconds of audio
    chunk = torch.ones(33075)  # 1.5 seconds
    buffer.add_chunk(1000.0, chunk)
    
    # Pop 1 second
    segment = buffer.pop_segment(1.0)
    assert len(segment) == 22050
    assert torch.all(segment == 1)
    assert len(buffer.chunks) == 1
    assert len(buffer.chunks[0][1]) == 11025  # 0.5 seconds remaining

def test_overlapping_chunks(buffer):
    # Add two overlapping chunks
    chunk1 = torch.ones(22050)
    chunk2 = torch.ones(22050) * 2
    buffer.add_chunk(1000.0, chunk1)  # 1 second
    buffer.add_chunk(1000.5, chunk2)  # Starting 0.5 seconds later
    
    segment = buffer.get_segment(1.5)
    assert len(segment) == 33075
    # First half should be ones, second half should be twos
    assert torch.all(segment[:11025] == 1)
    assert torch.all(segment[11025:] == 2)
