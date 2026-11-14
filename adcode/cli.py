"""Console entry point for AdCode."""

from __future__ import annotations

import sys


def main() -> int:
    from adcode.tui.app import AdCodeApp

    try:
        app = AdCodeApp()
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
