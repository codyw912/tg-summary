"""tg-summary-eval: evaluation tools, kept out of the main CLI.

- compare: run summarize/digest variants (settings, models, servers) on one window and write metrics as JSON.
- judge: grade digests saved by earlier compare runs with one judge, combining results files.

Neither prints or writes chat text; digests and judge evidence stay in the chat database.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tg_summary.cli import add_window_args, interrupted, keep_awake, load_window
from tg_summary.compare import run_compare, run_judge
from tg_summary.config import load_config
from tg_summary.llm import LLMError
from tg_summary.store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tg-summary-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="run role variants side by side; metrics only, as JSON")
    add_window_args(compare)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--variants", type=Path, required=True, help="TOML file of [[variant]] tables")
    compare.add_argument("--repeats", type=int, default=3, help="independent runs per variant (cache bypassed)")
    compare.add_argument("--no-judge", action="store_true", help="skip claim checks even if roles.judge is set")
    compare.add_argument(
        "--resume", action="store_true", help="keep the finished runs already in --out and run only the rest"
    )

    judge = sub.add_parser("judge", help="grade digests saved by earlier compare runs and print one table")
    judge.add_argument("--db", type=Path, required=True)
    judge.add_argument("--config", type=Path, default=Path("config.toml"))
    judge.add_argument(
        "--results", type=Path, action="append", required=True, help="a compare results file; repeat to combine"
    )
    judge.add_argument("--out", type=Path, required=True, help="combined results with judge metrics (JSON)")

    args = parser.parse_args(argv)
    try:
        with keep_awake():
            if args.command == "compare":
                # Comparisons reuse stored media and link results and never run that stage themselves.
                w = load_window(args, offline=True)
                return run_compare(args, w.config, w.store, w.messages, w.redactor_factory, w.derived)
            if not args.db.exists():
                raise FileNotFoundError(f"{args.db} does not exist")
            return run_judge(args, load_config(args.config), Store(args.db))
    except (ValueError, LLMError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if args.command == "compare":
            interrupted("finished runs are saved in --out; rerun with --resume to continue")
        interrupted("finished verdicts are saved, so rerunning the same command resumes")


if __name__ == "__main__":
    sys.exit(main())
