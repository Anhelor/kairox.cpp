#!/usr/bin/env python3
"""Summarize KAIROX benchmark logs as Markdown and optional CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUMMARY_RE = re.compile(r"^\s*benchmark summary:\s*$", re.MULTILINE)
DECODE_MEAN_RE = re.compile(r"^\s*decode mean\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s\s*$", re.MULTILINE)


@dataclass
class BenchRow:
    path: Path
    benchmark_group: str
    backend: str
    profile_or_hardware: str
    generation_mode: str
    model: str
    decode_mean: float | None
    neuralink_over_llama_cpp: float | None = None
    kairox_over_llama_cpp: float | None = None


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse KAIROX benchmark logs and emit a Markdown summary table.",
    )
    parser.add_argument("paths", nargs="+", help="Log files or directories containing .log files")
    parser.add_argument("--out", help="Write the Markdown table to this file")
    parser.add_argument("--csv", dest="csv_path", help="Write parsed rows to this CSV file")
    return parser.parse_args()


def iter_log_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            warn(f"path does not exist: {raw_path}")
            continue

        candidates = sorted(path.rglob("*.log")) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(candidate)

    return files


def parse_filename(path: Path) -> tuple[str, str, str, str, str]:
    parts = path.name.removesuffix(".log").split("__")
    benchmark_group = parts[0] if len(parts) >= 1 and parts[0] else "unknown"
    backend = parts[1] if len(parts) >= 2 and parts[1] else "unknown"
    profile = parts[2] if len(parts) >= 3 and parts[2] else "unknown"
    mode = parts[3] if len(parts) >= 4 and parts[3] else "unknown"
    model = parts[-1] if len(parts) >= 5 and parts[-1] else "unknown"
    return benchmark_group, backend, profile, mode, model


def parse_summary_value(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def parse_log(path: Path) -> BenchRow | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warn(f"cannot read {path}: {exc}")
        return None

    summaries = list(SUMMARY_RE.finditer(text))
    if not summaries:
        warn(f"no benchmark summary found: {path}")
        return None

    summary_text = text[summaries[-1].start() :]
    decode_mean = parse_summary_value(DECODE_MEAN_RE, summary_text)

    benchmark_group, backend, profile, mode, model = parse_filename(path)
    return BenchRow(
        path=path,
        benchmark_group=benchmark_group,
        backend=backend,
        profile_or_hardware=profile,
        generation_mode=mode,
        model=model,
        decode_mean=float(decode_mean) if decode_mean is not None else None,
    )


def speedup_key(row: BenchRow) -> tuple[str, str, str, str]:
    return (
        row.benchmark_group,
        row.profile_or_hardware,
        row.generation_mode,
        row.model,
    )


def add_llama_cpp_speedups(rows: list[BenchRow]) -> None:
    baselines: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in rows:
        if row.decode_mean is None:
            continue
        baselines.setdefault(speedup_key(row), {})[row.backend] = row.decode_mean

    for row in rows:
        row_baselines = baselines.get(speedup_key(row), {})
        llama_cpp = row_baselines.get("llama_cpp")
        if not llama_cpp:
            continue

        neuralink = row_baselines.get("neuralink")
        kairox = row_baselines.get("kairox")
        if neuralink is not None:
            row.neuralink_over_llama_cpp = neuralink / llama_cpp
        if kairox is not None:
            row.kairox_over_llama_cpp = kairox / llama_cpp


def sort_rows(rows: list[BenchRow]) -> list[BenchRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.profile_or_hardware,
            row.benchmark_group,
            row.generation_mode,
            row.model,
            row.backend,
        ),
    )


def fmt_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def fmt_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def md_escape(value: str) -> str:
    return value.replace("|", r"\|") if value else "-"


def render_markdown(rows: list[BenchRow]) -> str:
    headers = [
        "Benchmark Group",
        "Backend",
        "Profile/Hardware",
        "Mode",
        "Model",
        "Decode Mean (t/s)",
        "neuralink / llama_cpp",
        "kairox / llama_cpp",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        fields = [
            md_escape(row.benchmark_group),
            md_escape(row.backend),
            md_escape(row.profile_or_hardware),
            md_escape(row.generation_mode),
            md_escape(row.model),
            fmt_float(row.decode_mean),
            fmt_speedup(row.neuralink_over_llama_cpp),
            fmt_speedup(row.kairox_over_llama_cpp),
        ]
        lines.append("| " + " | ".join(fields) + " |")

    return "\n".join(lines) + "\n"


def write_csv(rows: list[BenchRow], csv_path: str) -> None:
    fieldnames = [
        "benchmark_group",
        "backend",
        "profile_or_hardware",
        "generation_mode",
        "model",
        "decode_mean_tps",
        "neuralink_over_llama_cpp",
        "kairox_over_llama_cpp",
        "path",
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "benchmark_group": row.benchmark_group,
                    "backend": row.backend,
                    "profile_or_hardware": row.profile_or_hardware,
                    "generation_mode": row.generation_mode,
                    "model": row.model,
                    "decode_mean_tps": "" if row.decode_mean is None else f"{row.decode_mean:.2f}",
                    "neuralink_over_llama_cpp": ""
                    if row.neuralink_over_llama_cpp is None
                    else f"{row.neuralink_over_llama_cpp:.2f}",
                    "kairox_over_llama_cpp": ""
                    if row.kairox_over_llama_cpp is None
                    else f"{row.kairox_over_llama_cpp:.2f}",
                    "path": str(row.path),
                }
            )


def main() -> int:
    args = parse_args()
    rows = [row for path in iter_log_files(args.paths) if (row := parse_log(path)) is not None]

    if not rows:
        print("error: no valid benchmark logs found", file=sys.stderr)
        return 1

    rows = sort_rows(rows)
    add_llama_cpp_speedups(rows)
    markdown = render_markdown(rows)
    print(markdown, end="")

    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")

    if args.csv_path:
        write_csv(rows, args.csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
