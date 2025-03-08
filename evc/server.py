import os
import tornado.ioloop
import tornado.web
import socketio
import json
import asyncio
from datetime import datetime
from collections import deque

class AppHandler(socketio.AsyncNamespace):
    def __init__(self, args):
        self.loop = asyncio.get_event_loop()
        self.port = args['port']
        super(AppHandler, self).__init__('/')

    def get_state(self):
        return {}

    async def send_state(self):
        await self.emit('state', {})

    def send_state_sync(self):
        self.do_sync(lambda: self.send_state())

    def do_sync(self, f):
        tornado.ioloop.IOLoop.current().add_callback(f)

    async def background_tick(self):
        while True:
            await asyncio.sleep(1)
            # self.send_state_sync()

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

def main(args):
    print("loading...")

    sio = socketio.AsyncServer(async_mode='tornado', cors_allowed_origins="*")
    apphandler = AppHandler(args)
    sio.register_namespace(apphandler)
    app = tornado.web.Application(
        [
            (r"/", TornadoHandler, {'apphandler': apphandler, 'args': args}),
            (r"/socket.io/", socketio.get_tornado_handler(sio)),
        ],
        template_path=os.path.dirname(__file__),
        static_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
        debug=True,
    )
    app.listen(args['port'])
    sio.start_background_task(apphandler.background_tick)
    print("running on port", args['port'])
    print("http://localhost:" + str(args['port']))
    tornado.ioloop.IOLoop.current().start()
