"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from . import config as cfgmod


def _cmd_presets(_: argparse.Namespace) -> int:
    for name in cfgmod.available_presets():
        cfg = cfgmod.load(name)
        print(f"{name:<16} {cfg.hash()}  {cfg.meta.notes}")
    return 0


def _cmd_show_config(args: argparse.Namespace) -> int:
    cfg = cfgmod.load(args.preset)
    if args.set:
        overrides: dict[str, object] = {}
        for item in args.set:
            key, _, raw = item.partition("=")
            if not _:
                raise SystemExit(f"--set expects key=value, got {item!r}")
            overrides[key.strip()] = json.loads(raw)
        cfg = cfg.replace(**overrides)
    print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    print(f"\n# config hash: {cfg.hash()}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vfp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("presets", help="list available presets with their config hash")
    p.set_defaults(func=_cmd_presets)

    p = sub.add_parser("show-config", help="resolve a preset and print it as JSON")
    p.add_argument("preset", help="preset name or path to a .toml file")
    p.add_argument(
        "--set",
        action="append",
        metavar="PATH=JSON",
        help="dotted-path override, e.g. --set numerics.dx_m=0.05 (repeatable)",
    )
    p.set_defaults(func=_cmd_show_config)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except cfgmod.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
