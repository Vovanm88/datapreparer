from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_builder
from .shards import discover_commoncatalog_shards
from .validate import validate_output


def main() -> None:
    parser = argparse.ArgumentParser(prog="prepare-sft-dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--limit-shards", type=int)

    inspect_parser = sub.add_parser("inspect-commoncatalog")
    inspect_parser.add_argument("--config", default="configs/commoncatalog_sft.yaml")
    inspect_parser.add_argument("--limit-shards", type=int, default=20)

    validate_parser = sub.add_parser("validate-output")
    validate_parser.add_argument("--dataset-root", required=True)

    args = parser.parse_args()
    if args.command == "run":
        cfg = load_config(args.config)
        summary = asyncio.run(run_builder(cfg, dry_run=args.dry_run, limit_shards=args.limit_shards))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "inspect-commoncatalog":
        cfg = load_config(args.config)
        shards = discover_commoncatalog_shards(cfg.dataset, limit=args.limit_shards)
        for shard in shards:
            print(shard.uri)
        print(f"candidate_shards={len(shards)}")
    elif args.command == "validate-output":
        errors = validate_output(Path(args.dataset_root))
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
        print("OK")
