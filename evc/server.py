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
import glob
from typing import Dict, List, Callable
from collections import defaultdict

PLAYBACK_DELAY = 10  # seconds
CHUNK_SIZE = 22050 * 5  # 5 seconds at 22050 Hz


class EventEmitter:
    def __init__(self):
        self._listeners = defaultdict(list)

    def on(self, event: str, callback: Callable):
        self._listeners[event].append(callback)

    def emit(self, event: str, *args, **kwargs):
        for callback in self._listeners[event]:
            callback(*args, **kwargs)


class AppHandler(socketio.AsyncNamespace):
    def __init__(self, args):
        self.loop = asyncio.get_event_loop()
        self.port = args["port"]
        self.buffer = ReceiveBuffer()  # Create buffer first
        self.streamer = Streamer(self.buffer)  # Pass buffer to streamer
        self.sample_rate = 22050
        self.min_sample_diff = None  # Minimum observed time difference
        self.start_time = None
        self.last_stats_time = time.time()
        super(AppHandler, self).__init__("/")

    async def on_audio_data(self, sid, data):
        server_time = time.time()
        if self.start_time is None:
            self.start_time = time.time()
        server_samples = (server_time - self.start_time) * self.sample_rate
        current_sample_diff = server_samples - data["offset"]
        self.min_sample_diff = min(
            self.min_sample_diff or float("inf"), current_sample_diff
        )

        audio_chunk = torch.tensor(data["data"], dtype=torch.float32)
        self.buffer.add_audio(audio_chunk, data["offset"])

        # Print stats periodically
        if server_time - self.last_stats_time >= 5.0:
            estimated_client_time = (
                time.time() + self.min_sample_diff / self.sample_rate
            )
            time_behind = estimated_client_time - server_time
            print(f"Input network delay: {time_behind:.3f}s")
            self.last_stats_time = server_time

    def get_state(self):
        return {}

    async def send_state(self):
        await self.emit("state", {})

    def send_state_sync(self):
        self.do_sync(lambda: self.send_state())

    def do_sync(self, f):
        tornado.ioloop.IOLoop.current().add_callback(f)

    async def on_command(self, sid, data):
        self.emit("command", data)

    async def on_connect(self, sid, environ):
        await self.send_state()
        return

    def on_disconnect(self, sid, reason):
        if hasattr(self, "streamer"):
            self.streamer.cleanup()


class TornadoHandler(tornado.web.RequestHandler):
    def initialize(self, apphandler, args):
        self.apphandler = apphandler
        self.args = args

    def get(self):
        self.render(
            "../static/index.html",
            debug=1,
            initial_state=json.dumps({}),
            port=self.args["port"],
        )


class ReceiveBuffer(EventEmitter):
    def __init__(self, sample_rate=22050, segment_duration=1, buffer_delay=2):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.buffer_delay = buffer_delay
        self.chunks = []  # list of (offset, chunk) tuples
        self.last_yielded_offset = None  # last yielded sample offset
        self.first_yielded_time = None
        self.total_yielded = 0
        self.lock = threading.Lock()
        self.process_thread = threading.Thread(
            target=self._process_segments, daemon=True
        )
        self.process_thread.start()

    def last_yielded_time(self):
        """Return the time of the end of the last yielded sample."""
        if self.first_yielded_time is None:
            return time.time()
        return self.first_yielded_time + self.total_yielded / self.sample_rate

    def add_audio(self, chunk, offset):
        """Add a new audio chunk with its offset to the buffer."""
        with self.lock:
            if (
                self.last_yielded_offset is not None
                and offset < self.last_yielded_offset
            ):
                # Drop chunk as it is too old
                return
            # If first chunk, initialize our starting time.
            if self.first_yielded_time is None:
                self.first_yielded_time = time.time()
            self.chunks.append((offset, chunk))

    def _process_segments(self):
        while True:
            time.sleep(0.01)
            if self.last_yielded_time() + self.buffer_delay > time.time():
                continue

            with self.lock:
                if self.last_yielded_offset is None:
                    self.last_yielded_offset = min(offset for offset, _ in self.chunks)

            start_offset = self.last_yielded_offset
            nominal_end_offset = start_offset + (
                self.segment_duration * self.sample_rate
            )

            with self.lock:
                selected = []
                remaining = []
                for offset, chunk in self.chunks:
                    if offset < nominal_end_offset:
                        selected.append((offset, chunk))
                    else:
                        remaining.append((offset, chunk))
                self.chunks = remaining

            if not selected:
                end_offset = nominal_end_offset
            else:
                end_offset = max(offset + len(chunk) for offset, chunk in selected)

            total_samples = end_offset - start_offset
            segment_buffer = torch.zeros(total_samples, dtype=torch.float32)

            for offset, chunk in selected:
                chunk_offset = offset - start_offset
                end_pos = chunk_offset + len(chunk)
                segment_buffer[chunk_offset:end_pos] = chunk

            segment_to_send = (
                (segment_buffer * 32767).clamp(-32768, 32767).to(torch.int16)
            )
            # Instead of directly writing to ffmpeg, emit an event
            self.emit(
                "audio_segment",
                {
                    "start_time": self.first_yielded_time
                    + start_offset / self.sample_rate,
                    "audio": segment_to_send.numpy().tobytes(),
                },
            )

            with self.lock:
                self.last_yielded_offset = end_offset
                self.total_yielded += total_samples


class ChunkProcessor(EventEmitter):
    def __init__(self, sample_rate=22050):
        super().__init__()
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=5)  # Store last 5 seconds
        self.running = True
        self.lock = threading.Lock()

        # Start processor thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

    def handle_audio_segment(self, segment_dict):
        """Receive 1-second segments from ReceiveBuffer"""
        audio_data = np.frombuffer(segment_dict["audio"], dtype=np.int16)
        with self.lock:
            self.buffer.append(audio_data)
            while len(self.buffer) > 5:
                self.buffer.popleft()

    def _process_loop(self):
        """Process 5-second overlapping windows"""
        while self.running:
            with self.lock:
                if len(self.buffer) >= 5:
                    # Concatenate last 5 seconds
                    full_chunk = np.concatenate(list(self.buffer))
                    self.buffer.popleft()

                    # Simulate intense processing (~0.5 sec)
                    processed_chunk = self._process_chunk(full_chunk)

                    # Forward only the last 1 second
                    last_second = processed_chunk[-len(self.buffer[-1]) :]
                    self.emit(
                        "processed_audio",
                        {"audio": last_second.tobytes()},
                    )

            time.sleep(0.01)  # Don't burn CPU

    def _process_chunk(self, chunk):
        """Simulate intensive processing"""
        # Simulate 0.5 seconds of heavy computation
        # time.sleep(0.5)

        # Do some dummy processing
        chunk = chunk.astype(np.float32)
        chunk = np.clip(chunk * 1.1, -32768, 32767).astype(np.int16)
        return chunk

    def cleanup(self):
        self.running = False
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)


class Streamer:
    def __init__(self, buffer: ReceiveBuffer):
        self.HLS_DIR = "hls_stream"
        self.ffmpeg_process = self.start_rtp_stream()
        self.input_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.running = True

        # Create processor
        self.processor = ChunkProcessor()
        buffer.on("audio_segment", self.processor.handle_audio_segment)
        self.processor.on("processed_audio", self.handle_audio_segment)

        # Start writer thread
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()

    def handle_audio_segment(self, segment_dict: Dict):
        """Just append to buffer instead of writing directly"""
        with self.buffer_lock:
            self.input_buffer.append(segment_dict["audio"])

    def _write_loop(self):
        """Continuously write buffered audio data to ffmpeg"""
        first_write = True
        while self.running:
            data_to_write = None
            with self.buffer_lock:
                if self.input_buffer:
                    data_to_write = self.input_buffer.popleft()

            if data_to_write is not None:
                if first_write:
                    time.sleep(1)
                    first_write = False
                try:
                    self.ffmpeg_process.stdin.write(data_to_write)
                    self.ffmpeg_process.stdin.flush()
                except (BrokenPipeError, IOError) as e:
                    print(f"FFmpeg pipe error: {e}")
                    self.running = False
                    break
            else:
                time.sleep(0.01)

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.processor.cleanup()
        if self.writer_thread.is_alive():
            self.writer_thread.join(timeout=1.0)
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=1.0)
            except Exception as e:
                self.ffmpeg_process.kill()

    def start_rtp_stream(self, sample_rate=22050, channels=1):
        ffmpeg_cmd = [
            "ffmpeg",
            "-re",
            # First input: a silent audio source
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=mono:sample_rate={sample_rate}",
            # Second input: piped audio
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            # Mix the two inputs: when no data comes on the pipe, the silent source fills in
            "-filter_complex",
            "[1:a]aresample=async=1,asetpts=PTS-STARTPTS[a1]; [0:a][a1]amix=inputs=2:duration=longest:dropout_transition=0",
            "-acodec",
            "aac",
            "-b:a",
            "128k",
            # (Optional reconnect options – note these are typically for network input)
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
            "-f",
            "rtp_mpegts",
            "rtp://127.0.0.1:1234",
        ]
        return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


def main(args):
    print("loading...")

    sio = socketio.AsyncServer(async_mode="tornado", cors_allowed_origins="*")
    apphandler = AppHandler(args)
    sio.register_namespace(apphandler)
    app = tornado.web.Application(
        [
            (r"/", TornadoHandler, {"apphandler": apphandler, "args": args}),
            (r"/socket.io/", socketio.get_tornado_handler(sio)),
            (r"/hls/(.*)", tornado.web.StaticFileHandler, {"path": "hls_stream"}),
        ],
        template_path=os.path.dirname(__file__),
        static_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
        debug=True,
    )
    app.listen(args["port"])
    print("running on port", args["port"])
    print("http://localhost:" + str(args["port"]))
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if hasattr(sio, "apphandler") and hasattr(sio.apphandler, "streamer"):
            sio.apphandler.streamer.cleanup()
