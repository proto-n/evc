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
from multiprocessing import Process, Queue

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
        self.audio_port = args["audio_port"]
        self.sample_rate = 22050
        self.is_running = False
        super(AppHandler, self).__init__("/")

    def initialize_components(self):
        """Initialize all components and variables"""
        self.buffer = ReceiveBuffer()
        self.streamer = Streamer(self.buffer, self.audio_port)
        self.min_sample_diff = None
        self.start_time = None
        self.last_stats_time = time.time()
        self.is_running = True

    def cleanup_components(self):
        """Cleanup all components and reset state"""
        self.is_running = False
        if hasattr(self, "streamer"):
            self.streamer.cleanup()
            delattr(self, "streamer")
        if hasattr(self, "buffer"):
            # Stop the buffer's processing thread
            if hasattr(self.buffer, "process_thread"):
                self.buffer.running = False
                self.buffer.process_thread.join(timeout=1.0)
            delattr(self, "buffer")
        self.min_sample_diff = None
        self.start_time = None
        self.last_stats_time = None

    async def on_audio_data(self, sid, data):
        if not self.is_running:
            return
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
        """Initialize everything on new connection"""
        self.initialize_components()
        await self.send_state()
        print("Client connected, initialized components")

    def on_disconnect(self, sid, reason):
        """Cleanup everything on disconnect"""
        print("Client disconnected, cleaning up components")
        self.cleanup_components()


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


def worker_process(task_queue, result_queue):
    import numpy as np

    while True:
        full_chunk, last_seg_len = task_queue.get()
        if full_chunk is None:  # Shutdown signal
            break
        # Dummy processing: scale and clip (simulate intensive processing)
        processed_chunk = full_chunk.astype(np.float32)
        processed_chunk = np.clip(processed_chunk * 1.1, -32768, 32767).astype(np.int16)
        result_queue.put((processed_chunk, last_seg_len))


class ChunkProcessor(EventEmitter):
    def __init__(self, sample_rate=22050):
        super().__init__()
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=5)  # Store last 5 seconds
        self.running = True
        self.lock = threading.Lock()
        self.crossfade_time = int(0.2 * sample_rate)  # 0.2 seconds crossfade
        self.previous_ending = None

        # Set up multiprocessing queues and start the worker process
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.worker_process = Process(
            target=worker_process, args=(self.task_queue, self.result_queue)
        )
        self.worker_process.start()

        # Start processor thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

    def handle_audio_segment(self, segment_dict):
        """Receive 1-second segments from ReceiveBuffer."""
        audio_data = np.frombuffer(segment_dict["audio"], dtype=np.int16)
        with self.lock:
            self.buffer.append(audio_data)
            while len(self.buffer) > 5:
                self.buffer.popleft()

    def _crossfade(
        self, chunk1: np.ndarray, chunk2: np.ndarray, overlap: int
    ) -> np.ndarray:
        """Apply crossfade between two audio chunks

        Args:
            chunk1: First audio chunk
            chunk2: Second audio chunk
            overlap: Number of samples to overlap

        Returns:
            Crossfaded audio chunk
        """
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2

        if len(chunk2) < overlap:
            chunk2[:overlap] = (
                chunk2[:overlap] * fade_in[: len(chunk2)]
                + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
            )
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    def _process_loop(self):
        """Process 5-second overlapping windows using multiprocessing queues."""
        while self.running:
            with self.lock:
                if len(self.buffer) >= 5:
                    # Concatenate last 5 seconds
                    full_chunk = np.concatenate(list(self.buffer))
                    # Slide the window by popping the oldest chunk
                    popped_chunk = self.buffer.popleft()
                    # Determine the length of the last segment (to forward only that part)
                    if self.buffer:
                        last_seg_len = len(self.buffer[-1])
                    else:
                        last_seg_len = len(popped_chunk)
                    # Put the full chunk in the task queue for processing
                    self.task_queue.put((full_chunk, last_seg_len))
            # Try to get the processed result (if available) with a short timeout.
            try:
                processed_chunk, last_seg_len = self.result_queue.get(timeout=0.1)
                # Extract the segment to emit (with extra samples for crossfade)
                current_segment = processed_chunk[
                    -last_seg_len - self.crossfade_time : -self.crossfade_time
                ]

                # Apply crossfade if we have a previous ending
                if self.previous_ending is not None:
                    current_segment = self._crossfade(
                        self.previous_ending, current_segment, self.crossfade_time
                    )

                # Store the ending for next crossfade
                self.previous_ending = processed_chunk[-self.crossfade_time :]

                self.emit("processed_audio", {"audio": current_segment.tobytes()})
            except Exception:
                # If no result is ready, continue looping
                pass

            time.sleep(0.01)  # Prevent high CPU usage

    def cleanup(self):
        self.running = False
        self.previous_ending = None
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        # Signal the worker process to exit and join it.
        self.task_queue.put(None)
        self.worker_process.join(timeout=1.0)


class Streamer:
    def __init__(self, buffer: ReceiveBuffer, audio_port: int):
        self.HLS_DIR = "hls_stream"
        self.audio_port = audio_port
        self.input_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.running = True
        self.ffmpeg_process = self.start_rtp_stream()

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
        # fmt: off
        ffmpeg_cmd = [
            "ffmpeg",
            "-re",
            "-vsync", "1",  # Synchronize timestamps properly

            # First input: silent audio source
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=mono:sample_rate={sample_rate}",

            # Second input: piped audio from stdin
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-stream_loop", "-1",
            # "-fflags", "+ignoreeof",  # Ignore EOF to prevent stopping
            "-i", "pipe:0",

            # Mix silent source with input; silence fills gaps when input stops
            "-filter_complex",
            "[1:a]aresample=async=1000,asetpts=PTS-STARTPTS[a1];"
            "[0:a][a1]amix=inputs=2:duration=longest:dropout_transition=0",

            "-acodec", "aac",
            "-b:a", "128k",

            # RTP streaming settings
            "-f", "rtp_mpegts",
            "rtp://127.0.0.1:" + str(self.audio_port),
        ]
        # fmt: on

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
