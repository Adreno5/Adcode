"""Command-line runner for the manual AdCode harness scenarios."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from bench.app import AppHarness, HarnessError
from bench.artifacts import ArtifactWriter
from bench.script import EchoResponder
from bench.scenarios import SCENARIOS, list_scenarios


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m bench',
        description='Run AdCode against a fake local LLM gateway and capture '
                    'screenshots of the real UI.')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('list', help='list the available scenarios')

    run = sub.add_parser('run', help='run one scenario (or all of them)')
    run.add_argument('scenario', nargs='?', default='all',
                     help='scenario name, or "all" (default)')
    run.add_argument('--out', default=None,
                     help='output directory (default: harness-output/<stamp>)')
    run.add_argument('--width', type=int, default=100,
                     help='simulated terminal columns (default 100)')
    run.add_argument('--height', type=int, default=30,
                     help='simulated terminal rows (default 30)')
    run.add_argument('--no-png', action='store_true',
                     help='write SVG screenshots only')
    run.add_argument('--raster-python', default=None,
                     help='interpreter to use for PNG rasterization')
    run.add_argument('--keep', action='store_true',
                     help='keep the temporary config/workspace tree')
    return parser


def _run_one(name: str, out_dir: Path, args: argparse.Namespace) -> Path:
    scenario = SCENARIOS[name]
    writer = ArtifactWriter(
        out_dir, png=not args.no_png, raster_python=args.raster_python)
    with AppHarness(width=args.width, height=args.height,
                    responder=EchoResponder(), keep=args.keep) as harness:
        harness.artifacts = writer
        scenario.run(harness)
        harness.snapshot('final')
        writer.finish(harness, name, scenario.summary)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == 'list':
        for scenario in list_scenarios():
            print(f'{scenario.name:<12} {scenario.summary}')
        return 0

    requested = args.scenario
    if requested == 'all':
        names = list(SCENARIOS)
    elif requested in SCENARIOS:
        names = [requested]
    else:
        print(f'Unknown scenario: {requested}', file=sys.stderr)
        print('Available: ' + ', '.join(SCENARIOS), file=sys.stderr)
        return 2

    base = Path(args.out or
                f'harness-output/{time.strftime("%Y%m%d-%H%M%S")}')
    failures: list[tuple[str, BaseException]] = []
    for name in names:
        out_dir = base if len(names) == 1 else base / name
        print(f'== {name} -> {out_dir}', flush=True)
        try:
            _run_one(name, out_dir, args)
        except HarnessError as error:
            failures.append((name, error))
            print(f'   FAILED: {error}', file=sys.stderr, flush=True)
        except Exception as error:  # keep going with the other scenarios
            failures.append((name, error))
            traceback.print_exc()
        else:
            print(f'   report: {out_dir / "report.md"}', flush=True)
    if failures:
        print(f'{len(failures)} scenario(s) failed', file=sys.stderr)
        return 1
    print(f'\nArtifacts: {base}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
