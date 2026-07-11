import argparse
import csv
import json
import re
import sys
from pathlib import Path


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
FIELD_RE = re.compile(rf"([A-Za-z0-9_]+)\s*:\s*({NUMBER_RE})")
ITER_RE = re.compile(r"iteration\s+(\d+)")
BEST_AVG_RE = re.compile(rf"val_best_avg_dice:\s*({NUMBER_RE})\s+at\s+(\d+)\s+iter")
NAMESPACE_RE = re.compile(r"Namespace\((.*)\)")
ARG_RE = re.compile(r"--([A-Za-z0-9_]+)(?:\s+([^-]\S*))?")


SUMMARY_COLUMNS = [
    "path",
    "dataset",
    "save_name",
    "model",
    "lb_domain",
    "lb_num",
    "domain_num",
    "max_iterations",
    "last_iteration",
    "last_epoch",
    "unet_best_avg_dice",
    "unet_best_avg_iter",
    "sam_best_avg_dice",
    "sam_best_avg_iter",
    "unet_final_avg_dice",
    "sam_final_avg_dice",
    "last_loss",
    "last_mask_ratio",
]


def parse_namespace(line):
    match = NAMESPACE_RE.search(line)
    if not match:
        return {}

    values = {}
    for part in split_top_level(match.group(1)):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = parse_value(value.strip())
    return values


def split_top_level(text):
    parts = []
    current = []
    quote = None
    depth = 0
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def parse_value(value):
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    if value in {"True", "False"}:
        return value == "True"
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_command(line):
    if "python train.py" not in line:
        return {}

    values = {}
    for key, value in ARG_RE.findall(line):
        values[key] = True if value == "" else value
    return values


def parse_metric_fields(line):
    return {key: float(value) for key, value in FIELD_RE.findall(line)}


def average_dice(metrics):
    dice_values = [
        value
        for key, value in metrics.items()
        if key.startswith("val_") and key.endswith("_dice")
    ]
    if not dice_values:
        return None
    return sum(dice_values) / len(dice_values)


def summarize_log(path, model_root):
    path = Path(path)
    model_root = Path(model_root)
    row = {
        "path": str(path.relative_to(model_root)) if path.is_relative_to(model_root) else str(path),
        "dataset": None,
        "save_name": None,
        "model": None,
        "lb_domain": None,
        "lb_num": None,
        "domain_num": None,
        "max_iterations": None,
        "last_iteration": None,
        "last_epoch": None,
        "unet_best_avg_dice": None,
        "unet_best_avg_iter": None,
        "sam_best_avg_dice": None,
        "sam_best_avg_iter": None,
        "unet_final_avg_dice": None,
        "sam_final_avg_dice": None,
        "last_loss": None,
        "last_mask_ratio": None,
    }

    current_model = None
    pending_epoch_model = None
    pending_epoch_metrics = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            command_values = parse_command(line)
            namespace_values = parse_namespace(line)
            for values in (command_values, namespace_values):
                for key in ("dataset", "save_name", "model", "lb_domain", "lb_num", "domain_num", "max_iterations"):
                    if key in values:
                        row[key] = values[key]

            if "iteration " in line:
                iter_match = ITER_RE.search(line)
                if iter_match:
                    row["last_iteration"] = int(iter_match.group(1))
                fields = parse_metric_fields(line)
                row["last_loss"] = fields.get("loss", row["last_loss"])
                row["last_mask_ratio"] = fields.get("mask_ratio", row["last_mask_ratio"])

            if "test unet model" in line:
                pending_epoch_model = None
                pending_epoch_metrics = None
                current_model = "unet"
                continue
            if "test sam model" in line:
                pending_epoch_model = None
                pending_epoch_metrics = None
                current_model = "sam"
                continue

            if " epoch " in line and "domain" not in line:
                epoch_match = re.search(r"epoch\s+(\d+)", line)
                if epoch_match:
                    row["last_epoch"] = int(epoch_match.group(1))
                    pending_epoch_model = current_model
                    pending_epoch_metrics = {}
                continue

            if pending_epoch_model and line.startswith("val_"):
                pending_epoch_metrics.update(parse_metric_fields(line))
                final_avg = average_dice(pending_epoch_metrics)
                if final_avg is not None:
                    row[f"{pending_epoch_model}_final_avg_dice"] = final_avg
                continue
            if pending_epoch_model:
                pending_epoch_model = None
                pending_epoch_metrics = None

            if "val_best_avg_dice:" in line:
                best_match = BEST_AVG_RE.search(line)
                if best_match and current_model:
                    prefix = "sam" if current_model == "sam" else "unet"
                    row[f"{prefix}_best_avg_dice"] = float(best_match.group(1))
                    row[f"{prefix}_best_avg_iter"] = int(best_match.group(2))

    return row


def find_logs(model_dir):
    return sorted(Path(model_dir).glob("**/log.txt"))


def write_csv(rows, output):
    writer = csv.DictWriter(output, fieldnames=SUMMARY_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: format_value(row.get(key)) for key in SUMMARY_COLUMNS})


def write_json(rows, output):
    json.dump(rows, output, indent=2)
    output.write("\n")


def write_markdown(rows, output):
    output.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
    output.write("| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |\n")
    for row in rows:
        output.write("| " + " | ".join(format_value(row.get(key)) for key in SUMMARY_COLUMNS) + " |\n")


def write_summary(rows, output, output_format):
    if output_format == "csv":
        write_csv(rows, output)
    elif output_format == "json":
        write_json(rows, output)
    else:
        write_markdown(rows, output)


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Summarize training results from model/**/log.txt files."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("../model"),
        help="Directory that contains dataset experiment folders. Default: ../model",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format. Default: markdown",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the summary to this file instead of printing it to stdout.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logs = find_logs(args.model_dir)
    rows = [summarize_log(path, args.model_dir) for path in logs]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as output:
            write_summary(rows, output, args.format)
    else:
        write_summary(rows, sys.stdout, args.format)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
