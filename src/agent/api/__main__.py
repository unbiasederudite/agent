"""CLI entrypoint for running the agent API server."""

import argparse
from pathlib import Path

import uvicorn

from agent.api.app import create_app


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, build the app, and start the uvicorn server."""
    parser = argparse.ArgumentParser(
        prog="python -m agent.api", description="Run the agent-core agent API."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the AppConfig JSON file."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    args = parser.parse_args(argv)

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
