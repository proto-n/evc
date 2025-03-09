import os
import tornado.ioloop
import tornado.web
import socketio
import json
import asyncio
import torch
import torchaudio
from datetime import datetime
from collections import deque
import subprocess
import time
import threading
import tornado.ioloop
import tornado.web
import numpy as np

PLAYBACK_DELAY = 10  # seconds
CHUNK_SIZE = 22050 * 5  # 5 seconds at 22050 Hz


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

class ChunkProcessor:
    def __init__(self, streamer):
        self.streamer = streamer
        self.current_buffer = []
        self.current_samples = 0
        self.processing_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.input_queue = deque()  # Queue of chunks waiting to be processed
        self.processing_thread.start()

    def add_audio(self, chunk, timestamp):
        self.current_buffer.append(chunk)
        self.current_samples += len(chunk)
        
        # If we have 5 seconds of audio, queue it for processing
        if self.current_samples >= CHUNK_SIZE:
            # Concatenate and split at exactly 5 seconds
            full_chunk = torch.cat(self.current_buffer)
            to_process = full_chunk[:CHUNK_SIZE]
            remainder = full_chunk[CHUNK_SIZE:]
            
            # Queue the 5-second chunk for processing
            self.input_queue.append((time.time(), to_process))
            
            # Reset buffer with remainder
            self.current_buffer = [remainder] if len(remainder) > 0 else []
            self.current_samples = len(remainder)

    def mock_deep_learning(self, audio):
        """Mock deep learning processing with 3-second delay"""
        time.sleep(3)
        # Simulate some processing (e.g., adding reverb)
        return audio * 0.8

    def process_loop(self):
        while True:
            if self.input_queue:
                timestamp, chunk = self.input_queue.popleft()
                print(f"Processing chunk recorded at {timestamp:.2f}")
                processed = self.mock_deep_learning(chunk)
                processed = (processed * 32767).to(torch.int16)
                self.streamer.add_audio(processed, timestamp)
            else:
                time.sleep(0.1)


class AppHandler(socketio.AsyncNamespace):
    def __init__(self, args):
        self.loop = asyncio.get_event_loop()
        self.port = args['port']
        self.streamer = Streamer()
        self.processor = ChunkProcessor(self.streamer)
        self.sample_rate = 22050
        self.min_time_diff = None  # Minimum observed time difference
        self.last_stats_time = time.time()
        super(AppHandler, self).__init__('/')

    async def on_audio_data(self, sid, data):
        server_time = time.time()
        audio_chunk = torch.tensor(data['data'], dtype=torch.float32)

        time_diff = server_time - data['clientTime']
        
        if self.min_time_diff is None or time_diff < self.min_time_diff:
            self.min_time_diff = time_diff
            print(f"Updated minimum time difference: {self.min_time_diff:.3f}s")

        estimated_client_time = server_time + self.min_time_diff
        current_delay = estimated_client_time - data['clientTime']
        
        if server_time - self.last_stats_time >= 5.0:
            print(f"Current delay: {current_delay*1000:.1f}ms")
            self.last_stats_time = server_time

        self.processor.add_audio(audio_chunk, estimated_client_time)

    def get_state(self):
        return {}

    async def send_state(self):
        await self.emit('state', {})

    def send_state_sync(self):
        self.do_sync(lambda: self.send_state())

    def do_sync(self, f):
        tornado.ioloop.IOLoop.current().add_callback(f)

    async def on_command(self, sid, data):
        self.emit('command', data)

    async def on_connect(self, sid, environ):
        await self.send_state()
        return 

    def on_disconnect(self, sid, reason):
        pass

class TornadoHandler(tornado.web.RequestHandler):
    def initialize(self, apphandler, args):
        self.apphandler = apphandler
        self.args = args

    def get(self):
        self.render(
            "../static/index.html",
            debug=1,
            initial_state=json.dumps({}),
            port=self.args['port']
        )

class Streamer:
    def __init__(self):
        self.HLS_DIR = 'hls_stream'
        os.makedirs(self.HLS_DIR, exist_ok=True)
        self.buffer = []  # [(timestamp, chunk), ...]
        self.ffmpeg_process = self.start_hls_stream()
        self.feed_thread = threading.Thread(target=self.feed_audio_loop, daemon=True)
        self.feed_thread.start()

    def add_audio(self, chunk, timestamp):
        """Modified to accept timestamp with chunk"""
        self.buffer.append((timestamp, chunk))

    def get_chunk_to_play(self):
        """Get the most recent chunk that's ready to play, drop older chunks"""
        if not self.buffer:
            return None

        current_time = time.time()
        target_time = current_time - PLAYBACK_DELAY
        
        # Find the last chunk that should be played
        play_idx = -1
        for i, (timestamp, _) in enumerate(self.buffer):
            if timestamp <= target_time:
                play_idx = i
            else:
                break

        if play_idx == -1:
            return None

        # Get the last valid chunk and remove all older chunks
        _, chunk_to_play = self.buffer[play_idx]
        self.buffer = self.buffer[play_idx + 1:]
        return chunk_to_play

    def flush_buffer(self):
        chunk = self.get_chunk_to_play()
        if chunk is None:
            return
        
        try:
            print(f"Playing chunk of size: {len(chunk)}, buffer size: {len(self.buffer)}")
            self.ffmpeg_process.stdin.write(chunk.numpy().tobytes())
            self.ffmpeg_process.stdin.flush()
        except Exception as e:
            print("Error writing to FFmpeg stdin:", e)

    def feed_audio_loop(self):
        while True:
            self.flush_buffer()
            time.sleep(0.1)  # Check every 100ms

    def start_hls_stream(self, sample_rate=22050, channels=1):
        output_path = os.path.join(self.HLS_DIR, "stream.m3u8")
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-i", "pipe:0",
            "-acodec", "aac",
            "-b:a", "128k",
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "5",
            "-hls_flags", "delete_segments",
            output_path
        ]
        return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

def main(args):
    print("loading...")

    sio = socketio.AsyncServer(async_mode='tornado', cors_allowed_origins="*")
    apphandler = AppHandler(args)
    sio.register_namespace(apphandler)
    app = tornado.web.Application(
        [
            (r"/", TornadoHandler, {'apphandler': apphandler, 'args': args}),
            (r"/socket.io/", socketio.get_tornado_handler(sio)),
            (r"/hls/(.*)", tornado.web.StaticFileHandler, {"path": "hls_stream"}),
        ],
        template_path=os.path.dirname(__file__),
        static_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
        debug=True,
    )
    app.listen(args['port'])
    print("running on port", args['port'])
    print("http://localhost:" + str(args['port']))
    tornado.ioloop.IOLoop.current().start()
