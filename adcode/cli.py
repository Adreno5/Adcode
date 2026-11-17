"""Console entry point for AdCode."""

from __future__ import annotations

import argparse
import secrets
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='adcode',
        description='A configurable full-screen terminal coding agent.',
    )
    parser.add_argument(
        '--web', action='store_true',
        help='serve the terminal UI to browsers on the local network')
    parser.add_argument(
        '--host', default='0.0.0.0',
        help='address the web UI listens on (default: %(default)s)')
    parser.add_argument(
        '--port', type=int, default=8765,
        help='port the web UI listens on (default: %(default)s)')
    parser.add_argument(
        '--token', default=None,
        help='shared secret required in the page URL '
             '(default: random, printed with the URL)')
    parser.add_argument(
        '--no-token', action='store_true',
        help='serve without a token: anyone who can reach the port can drive '
             'the session')
    parser.add_argument(
        '--no-resume', action='store_true',
        help='start a new conversation instead of resuming the last one')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.web:
        from adcode.web.server import run_web
        token = None if args.no_token else (args.token or secrets.token_hex(4))
        try:
            return run_web(host=args.host, port=args.port, token=token,
                           resume=not args.no_resume)
        except KeyboardInterrupt:
            return 130
        except Exception as error:
            print(f'AdCode web failed to start: {error}', file=sys.stderr)
            return 1

    from adcode.tui.app import AdCodeApp

    try:
        app = AdCodeApp(resume=not args.no_resume)
    except Exception as error:
        print(f'AdCode failed to start: {error}', file=sys.stderr)
        return 1
    try:
        app.run()
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'AdCode crashed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
