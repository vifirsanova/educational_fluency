#!/usr/bin/env python3

"""
Calculate descriptive statistics for the raw source corpus.

Supported input formats:
1. A directory containing .json and/or .jsonl files.
2. A single JSON file containing a list of records.
3. A wrapper structure such as:
   {
       "raw data": {
           "arxiv_texts.json": [...],
           "pubmed_texts.json": [...]
       }
   }
4. JSONL files containing one record per line.

A raw record is expected to contain:
- source
- category
- title
- text

Example:
{
    "source": "arxiv",
    "category": "q-bio",
    "title": "...",
    "text": "..."
}
"""

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_FIELDS = ("source", "category", "title", "text")


def normalize_space(value: Any) -> str:
    """Convert a value to normalized text."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized_text_hash(text: str) -> str:
    """Create a stable hash for duplicate detection."""
    normalized = normalize_space(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def count_words(text: str) -> int:
    """Count word-like units."""
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def count_sentences(text: str) -> int:
    """
    Approximate sentence count.

    This is intentionally dependency-free and should be described as an
    estimate rather than an exact linguistic sentence segmentation.
    """
    text = normalize_space(text)
    if not text:
        return 0

    parts = re.split(r"(?<=[.!?])\s+", text)
    return sum(1 for part in parts if part.strip())


def estimate_tokens(text: str) -> int:
    """Estimate tokens using characters / 4."""
    if not text:
        return 0
    return round(len(text) / 4)


def percentile(values: List[float], probability: float) -> float:
    """Calculate a percentile with linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered))
    fraction = position - lower_index

    lower = ordered[lower_index]
    upper = ordered[upper_index]

    return float(lower + (upper - lower) * fraction)


def summarize_values(values: List[int]) -> Dict[str, float]:
    """Return standard descriptive statistics."""
    if not values:
        return {
            "total": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "p25": 0,
            "p75": 0,
            "p95": 0,
            "max": 0,
        }

    return {
        "total": int(sum(values)),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "std": round(statistics.pstdev(values), 2),
        "min": int(min(values)),
        "p25": round(percentile(values, 0.25), 2),
        "p75": round(percentile(values, 0.75), 2),
        "p95": round(percentile(values, 0.95), 2),
        "max": int(max(values)),
    }


def looks_like_raw_record(obj: Any) -> bool:
    """Determine whether a dictionary is a raw-corpus record."""
    return (
        isinstance(obj, dict)
        and "text" in obj
        and any(field in obj for field in ("source", "category", "title"))
        and "hallucinated" not in obj
    )


def recursively_extract_records(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively extract raw records from lists and wrapper dictionaries.
    """
    if looks_like_raw_record(obj):
        yield obj
        return

    if isinstance(obj, list):
        for item in obj:
            yield from recursively_extract_records(item)

    elif isinstance(obj, dict):
        for value in obj.values():
            yield from recursively_extract_records(value)


def load_json_file(path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return list(recursively_extract_records(data))


def load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {error}"
                ) from error

            records.extend(recursively_extract_records(item))

    return records


def discover_files(input_path: Path) -> List[Path]:
    """Find JSON and JSONL files."""
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )

    if not files:
        raise FileNotFoundError(
            f"No .json or .jsonl files found under {input_path}"
        )

    return files


def clean_record(record: Dict[str, Any], input_file: str) -> Dict[str, Any]:
    """Normalize fields without modifying the original content."""
    return {
        "source": normalize_space(record.get("source")) or "unknown",
        "category": normalize_space(record.get("category")) or "unknown",
        "title": normalize_space(record.get("title")),
        "text": str(record.get("text") or "").strip(),
        "_input_file": input_file,
    }


def calculate_group_statistics(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate statistics for a subset of records."""
    characters = [len(record["text"]) for record in records]
    words = [count_words(record["text"]) for record in records]
    sentences = [count_sentences(record["text"]) for record in records]
    tokens = [estimate_tokens(record["text"]) for record in records]

    return {
        "records": len(records),
        "characters": summarize_values(characters),
        "words": summarize_values(words),
        "sentences_estimated": summarize_values(sentences),
        "tokens_estimated": summarize_values(tokens),
    }


def group_records(
    records: List[Dict[str, Any]],
    keys: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by one or more fields."""
    groups = defaultdict(list)

    for record in records:
        key = " / ".join(record[field] for field in keys)
        groups[key].append(record)

    return dict(groups)


def calculate_report(
    records: List[Dict[str, Any]],
    files: List[Path],
) -> Dict[str, Any]:
    """Calculate the complete corpus report."""
    missing_fields = Counter()
    empty_text_records = 0

    for record in records:
        for field in REQUIRED_FIELDS:
            if not normalize_space(record.get(field)):
                missing_fields[field] += 1

        if not record["text"].strip():
            empty_text_records += 1

    text_hashes = Counter(
        normalized_text_hash(record["text"])
        for record in records
        if record["text"].strip()
    )

    duplicate_groups = {
        text_hash: count
        for text_hash, count in text_hashes.items()
        if count > 1
    }

    duplicate_records_beyond_first = sum(
        count - 1 for count in duplicate_groups.values()
    )

    source_groups = group_records(records, ["source"])
    category_groups = group_records(records, ["category"])
    source_category_groups = group_records(records, ["source", "category"])

    return {
        "input_files": [str(path) for path in files],
        "overall": calculate_group_statistics(records),
        "quality_checks": {
            "empty_text_records": empty_text_records,
            "missing_fields": dict(sorted(missing_fields.items())),
            "unique_normalized_texts": len(text_hashes),
            "duplicate_text_groups": len(duplicate_groups),
            "duplicate_records_beyond_first": duplicate_records_beyond_first,
        },
        "by_source": {
            key: calculate_group_statistics(group)
            for key, group in sorted(source_groups.items())
        },
        "by_category": {
            key: calculate_group_statistics(group)
            for key, group in sorted(category_groups.items())
        },
        "by_source_and_category": {
            key: calculate_group_statistics(group)
            for key, group in sorted(source_category_groups.items())
        },
    }


def print_compact_table(report: Dict[str, Any]) -> None:
    """Print manuscript-friendly summary tables."""
    overall = report["overall"]

    print("\n=== OVERALL CORPUS ===")
    print(f"Records:            {overall['records']:,}")
    print(f"Characters:         {overall['characters']['total']:,}")
    print(f"Words:              {overall['words']['total']:,}")
    print(f"Estimated tokens:   {overall['tokens_estimated']['total']:,}")
    print(
        "Mean characters:    "
        f"{overall['characters']['mean']:,.2f}"
    )
    print(
        "Median characters:  "
        f"{overall['characters']['median']:,.2f}"
    )

    quality = report["quality_checks"]

    print("\n=== QUALITY CHECKS ===")
    print(f"Empty texts:                   {quality['empty_text_records']:,}")
    print(
        "Unique normalized texts:      "
        f"{quality['unique_normalized_texts']:,}"
    )
    print(
        "Duplicate text groups:        "
        f"{quality['duplicate_text_groups']:,}"
    )
    print(
        "Duplicate records after first: "
        f"{quality['duplicate_records_beyond_first']:,}"
    )
    print(f"Missing fields: {quality['missing_fields']}")

    print("\n=== STATISTICS BY SOURCE ===")
    header = (
        f"{'Source':<25}"
        f"{'Records':>10}"
        f"{'Characters':>15}"
        f"{'Words':>13}"
        f"{'Tokens est.':>15}"
        f"{'Mean chars':>13}"
    )
    print(header)
    print("-" * len(header))

    for source, stats in report["by_source"].items():
        print(
            f"{source:<25}"
            f"{stats['records']:>10,}"
            f"{stats['characters']['total']:>15,}"
            f"{stats['words']['total']:>13,}"
            f"{stats['tokens_estimated']['total']:>15,}"
            f"{stats['characters']['mean']:>13,.1f}"
        )

    print("\n=== COUNTS BY SOURCE AND CATEGORY ===")
    header = (
        f"{'Source / category':<50}"
        f"{'Records':>10}"
        f"{'Mean chars':>15}"
        f"{'Median chars':>17}"
    )
    print(header)
    print("-" * len(header))

    for group_name, stats in report["by_source_and_category"].items():
        print(
            f"{group_name:<50}"
            f"{stats['records']:>10,}"
            f"{stats['characters']['mean']:>15,.1f}"
            f"{stats['characters']['median']:>17,.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate raw-corpus statistics."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="A JSON/JSONL file or a directory containing corpus files.",
    )

    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the complete JSON report.",
    )

    args = parser.parse_args()

    files = discover_files(args.input)
    records = []

    print(f"Discovered {len(files)} input file(s).")

    for path in files:
        if path.suffix.lower() == ".jsonl":
            loaded = load_jsonl_file(path)
        else:
            loaded = load_json_file(path)

        print(f"Loaded {len(loaded):,} records from {path}")

        records.extend(
            clean_record(record, str(path))
            for record in loaded
        )

    if not records:
        raise ValueError("No raw-corpus records were found.")

    report = calculate_report(records, files)
    print_compact_table(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)

        with args.json_out.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        print(f"\nComplete report written to: {args.json_out}")


if __name__ == "__main__":
    main()
