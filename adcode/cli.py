"""Console entry point for AdCode."""

from __future__ import annotations

import argparse
import os
import secrets
import sys

from adcode.i18n import LOCALES


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
    parser.add_argument(
        '--locale', default=None, choices=list(LOCALES),
        help='language of the interface: en or zh '
             '(default: the one saved in the configuration)')
    return parser


def main(argv: list[str] | None = None) -> int:
    # The project comes first: the state directory is resolved against it,
    # and the log file is one of the things that lives there.  Without this
    # the log file would be opened before the app knows which directory it
    # is working in -- and it would land in the wrong place.
    from adcode import paths
    paths.set_project_root(os.getcwd())
    # Before anything else: the logs go to a file.  A stray warning on
    # stderr would land on the full-screen interface and destroy the frame.
    from adcode.logging_setup import configure_logging
    configure_logging()
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
    if args.locale:
        # The flag wins over the saved language for this run; the settings
        # browser can change it again.
        app.config.ui.locale = args.locale
        app.config_changed()
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
