"""Compare two saved evaluation JSON files."""
import argparse
from pathlib import Path

from grounded_docs.compare import compare_files

parser = argparse.ArgumentParser()
parser.add_argument("baseline", type=Path)
parser.add_argument("optimized", type=Path)
parser.add_argument("--output", type=Path, default=Path("data/evaluations/results/comparison.md"))
args = parser.parse_args()
print(compare_files(args.baseline, args.optimized, args.output)["markdown"])
