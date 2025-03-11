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

PLAYBACK_DELAY = 10  # seconds
CHUNK_SIZE = 22050 * 5  # 5 seconds at 22050 Hz


class AppHandler(socketio.AsyncNamespace):
    def __init__(self, args):
        self.loop = asyncio.get_event_loop()
        self.port = args['port']
        self.streamer = Streamer()
        self.buffer = ReceiveBuffer(self.streamer)
        self.sample_rate = 22050
        self.min_sample_diff = None  # Minimum observed time difference
        self.start_time = None
        self.last_stats_time = time.time()
        super(AppHandler, self).__init__('/')

    async def on_audio_data(self, sid, data):
        server_time = time.time()
        if self.start_time is None:
            self.start_time = time.time()
        server_samples = (server_time - self.start_time) * self.sample_rate
        current_sample_diff = server_samples - data['offset']
        self.min_sample_diff = min(self.min_sample_diff or float('inf'), current_sample_diff)

        audio_chunk = torch.tensor(data['data'], dtype=torch.float32)
        self.buffer.add_audio(audio_chunk, data['offset'])

        # Print stats periodically
        if server_time - self.last_stats_time >= 5.0:
            estimated_client_time = time.time() + self.min_sample_diff / self.sample_rate
            time_behind = estimated_client_time - server_time
            print(f"Input network delay: {time_behind:.3f}s")
            self.last_stats_time = server_time

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

class ReceiveBuffer:
    def __init__(self, streamer, sample_rate=22050, segment_duration=5, buffer_delay=10):
        self.streamer = streamer
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.buffer_delay = buffer_delay
        self.chunks = []  # list of (offset, chunk) tuples
        self.last_yielded_offset = None  # last yielded sample offset
        self.first_yielded_time = None
        self.total_yielded = 0
        self.lock = threading.Lock()
        self.process_thread = threading.Thread(target=self._process_segments, daemon=True)
        self.process_thread.start()

    def last_yielded_time(self):
        if self.first_yielded_time is None:
            self.first_yielded_time = time.time()
        return self.first_yielded_time + self.total_yielded / self.sample_rate

    def add_audio(self, chunk, offset):
        """Add a new audio chunk with its offset to the buffer."""
        with self.lock:
            if self.last_yielded_offset is not None and offset < self.last_yielded_offset:
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
            nominal_end_offset = start_offset + (self.segment_duration * self.sample_rate)

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

            segment_to_send = (segment_buffer * 32767).clamp(-32768, 32767).to(torch.int16)
            self.streamer.ffmpeg_process.stdin.write(segment_to_send.numpy().tobytes())
            self.streamer.ffmpeg_process.stdin.flush()

            with self.lock:
                self.last_yielded_offset = end_offset
                self.total_yielded += total_samples


class Streamer:
    def __init__(self):
        self.HLS_DIR = 'hls_stream'
        self.cleanup_hls_dir()
        self.buffer = []  # [(timestamp, chunk), ...]
        self.ffmpeg_process = self.start_hls_stream()
        # self.feed_thread = threading.Thread(target=self.feed_audio_loop, daemon=True)
        # self.feed_thread.start()

    def add_audio(self, chunk, timestamp):
        """Modified to accept timestamp with chunk"""
        self.buffer.append((timestamp, chunk))

    def cleanup_hls_dir(self):
        """Remove all existing HLS stream files"""
        if os.path.exists(self.HLS_DIR):
            for file in glob.glob(os.path.join(self.HLS_DIR, '*')):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error removing {file}: {e}")
        os.makedirs(self.HLS_DIR, exist_ok=True)

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
