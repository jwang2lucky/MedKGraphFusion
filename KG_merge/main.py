# main.py
import argparse
import logging
from merger import merge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def main():
    parser = argparse.ArgumentParser(description="Merge custom triples into primeKG")
    parser.add_argument("--kg",       default="kg.csv",
                        help="Path to primeKG kg.csv")
    parser.add_argument("--input",    default="input_triples.jsonl",
                        help="Path to input triples JSONL")
    parser.add_argument("--output",   default="kg_merged.csv",
                        help="Output merged CSV path")
    parser.add_argument("--save-extra", action="store_true",
                        help="Keep _origins/_support/_input_predicate columns")
    args = parser.parse_args()

    merge(
        kg_path=args.kg,
        input_path=args.input,
        output_path=args.output,
        save_extra_cols=args.save_extra,
    )

if __name__ == "__main__":
    main()