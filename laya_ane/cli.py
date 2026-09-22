"""Command-line prediction and CoreML checkpoint conversion."""

import argparse
import json
from pathlib import Path

from . import __version__
from .agent import DTYPES, Agent


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-ane", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "convert"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", default="convaiinnovations/laya")
        sub.add_argument("--subfolder")
        sub.add_argument("--revision")
        sub.add_argument("--dtype", choices=list(DTYPES), default="float16")
        if command == "predict":
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument("--state", help="Plain text input")
            source.add_argument("--state-file", type=Path, help="JSON state file")
            sub.add_argument(
                "--questions", required=True, type=Path, help="JSON question definitions"
            )
            sub.add_argument("--device", choices=("all", "ane", "gpu", "cpu"), default="all")
            sub.add_argument("--batch-size", type=int, default=16)
        else:
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--max-batch", type=int, default=8)
            sub.add_argument("--max-markers", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "convert":
        from .convert import convert

        result = convert(
            args.model,
            args.output,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
            max_batch=args.max_batch,
            max_markers=args.max_markers,
        )
        print(json.dumps({"output": str(result), "dtype": args.dtype, "backend": "coreml"}))
    else:
        state = args.state if args.state is not None else json.loads(args.state_file.read_text())
        questions = json.loads(args.questions.read_text())
        agent = Agent(
            args.model,
            device=args.device,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
            batch_size=args.batch_size,
        )
        print(json.dumps(agent.predict(state, questions), ensure_ascii=False, indent=2))
