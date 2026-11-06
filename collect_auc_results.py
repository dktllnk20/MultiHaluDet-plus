"""Collect the latest ensemble AUC for each model, dataset, and seed."""
import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_NAME = re.compile(
    r"^(?P<dataset>halueval|triviaqa)_(?P<variant>base|v2)_ensemble_s(?P<seed>\d+)\.log$"
)
LIMIT_LINE = re.compile(r"^Stage: ensemble \| seed: (\d+) \| limit: (\d+|all)$", re.MULTILINE)
AUC_LINE = re.compile(r"^AUC: ([0-9]+(?:\.[0-9]+)?)", re.MULTILINE)
CSV_FIELDS = ("model", "dataset", "seed", "base", "v2", "delta")


def _latest_auc(log_path, expected_seed, limit):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    starts = list(re.finditer(r"^Run started:.*$", text, re.MULTILINE))
    latest = None
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        section = text[start.start():end]
        metadata = LIMIT_LINE.search(section)
        auc = AUC_LINE.search(section)
        model = re.search(r"^Model: (.+)$", section, re.MULTILINE)
        if not metadata or not auc or not model:
            continue
        seed, run_limit = metadata.groups()
        if int(seed) != expected_seed or run_limit != str(limit):
            continue
        latest = (model.group(1).strip(), float(auc.group(1)))
    return latest


def collect_results(limit, logs_dir):
    results = {}
    for log_path in sorted(logs_dir.glob("*/*_ensemble_s*.log")):
        match = LOG_NAME.match(log_path.name)
        if not match:
            continue
        dataset, variant, seed_text = match.groups()
        seed = int(seed_text)
        result = _latest_auc(log_path, seed, limit)
        if result is None:
            continue
        model, auc = result
        key = (model, dataset, seed)
        results.setdefault(key, {})[variant] = auc

    rows = []
    for (model, dataset, seed), variants in sorted(results.items()):
        base = variants.get("base")
        v2 = variants.get("v2")
        delta = v2 - base if base is not None and v2 is not None else None
        rows.append({
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "base": "" if base is None else f"{base:.4f}",
            "v2": "" if v2 is None else f"{v2:.4f}",
            "delta": "" if delta is None else f"{delta:.4f}",
        })
    return rows


def _print_table(rows):
    display_rows = [
        [str(row[field]) if row[field] != "" else "N/A" for field in CSV_FIELDS]
        for row in rows
    ]
    widths = [max(len(field), *(len(row[index]) for row in display_rows))
              for index, field in enumerate(CSV_FIELDS)]
    header = " | ".join(field.upper().ljust(widths[index])
                        for index, field in enumerate(CSV_FIELDS))
    separator = "-+-".join("-" * width for width in widths)
    print(header)
    print(separator)
    for row in display_rows:
        print(" | ".join(value.ljust(widths[index])
                         for index, value in enumerate(row)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect the latest base and v2 ensemble AUCs for a dataset limit."
    )
    parser.add_argument("--limit", default="all",
                        help="Only include this record limit (default: all).")
    parser.add_argument("--output", type=Path,
                        help="CSV output path (default: results/auc_results_limit<LIMIT>.csv).")
    args = parser.parse_args(argv)
    if args.limit != "all":
        try:
            limit = int(args.limit)
        except ValueError:
            parser.error("--limit must be 'all' or a positive integer.")
        if limit < 1:
            parser.error("--limit must be 'all' or a positive integer.")
    else:
        limit = "all"

    rows = collect_results(limit, PROJECT_ROOT / "logs")
    output_path = args.output or PROJECT_ROOT / "results" / f"auc_results_limit{limit}.csv"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        _print_table(rows)
    else:
        print(f"No ensemble AUC results found for limit {limit}.")
        print("model,dataset,seed,base,v2,delta")
    try:
        display_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output_path
    print(f"Saved {len(rows)} row(s) to {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())