from .server import main
import argparse

parser = argparse.ArgumentParser(description="Run demo")
parser.add_argument(
    "--port",
    metavar="[port]",
    type=int,
    nargs="?",
    default=5000,
    help="port to listen on",
)
parser.add_argument(
    "--audio_port",
    metavar="[port]",
    type=int,
    nargs="?",
    default=1234,
    help="port to listen on",
)
parser.add_argument(
    "--stream_type",
    metavar="[port]",
    type=str,
    nargs="?",
    default="rtp",
    help="type of stream",
)

args = vars(parser.parse_args())
main(args)
