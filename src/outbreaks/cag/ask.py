from __future__ import annotations

import argparse

from outbreaks.cag.engine import OutbreaksCAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the Outbreaks CAG assistant.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--region", default=None, help="Optional region key (without .md).")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cag = OutbreaksCAG(max_new_tokens=args.max_new_tokens)
    base_path = cag.knowledge_dir / "playbooks" / "general.md"
    if not base_path.exists():
        raise SystemExit(f"Missing base knowledge file: {base_path}")
    cag.load_model()
    cag.build_base_cache(base_path.read_text(encoding="utf-8"))
    answer = cag.ask(args.question, region_key=args.region)
    print(answer)


if __name__ == "__main__":
    main()
