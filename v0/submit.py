#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List

DEFAULT_LRS = ["2e-4", "1e-4", "8e-5", "5e-5"]
DEFAULT_EPOCHS = ["10", "20"]


def normalize_list(values: List[str]) -> List[str]:
    if len(values) == 1 and "," in values[0]:
        return [v.strip() for v in values[0].split(",") if v.strip()]
    return values


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def build_job_script(
    script_path: Path,
    job_path: Path,
    lr: str,
    ep: str,
    run_name: str,
    output_dir: str,
    directives: List[str],
    script_args: List[str],
) -> None:
    args = shlex.join(script_args)
    content = (
        "#!/usr/bin/env bash\n"
        + "".join(directives)
        "set -euo pipefail\n"
        f'export LR="{lr}"\n'
        f'export EPOCHS="{ep}"\n'
        f'export RUN_NAME="{run_name}"\n'
        f'export OUTPUT_DIR="{output_dir}"\n'
        f'exec bash "{script_path}"{(" " + args) if args else ""}\n'
    )
    job_path.write_text(content)
    job_path.chmod(0o755)


def extract_scheduler_directives(script_path: Path) -> List[str]:
    if not script_path.exists():
        return []
    directives = []
    pattern = re.compile(r"^#(?:SBATCH|PBS|BSUB|\\$)")
    for line in script_path.read_text().splitlines(keepends=True):
        if pattern.match(line):
            directives.append(line if line.endswith("\n") else f"{line}\n")
    return directives


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit grid jobs for a training shell script."
    )
    parser.add_argument("script", help="Path to the training shell script")
    parser.add_argument(
        "--lr",
        nargs="+",
        default=DEFAULT_LRS,
        help="Learning rates (space or comma separated)",
    )
    parser.add_argument(
        "--epochs",
        "--ep",
        dest="epochs",
        nargs="+",
        default=DEFAULT_EPOCHS,
        help="Epoch counts (space or comma separated)",
    )
    parser.add_argument(
        "--submit-cmd",
        default=os.environ.get("SUBMIT_CMD", "bash"),
        help="Command used to submit jobs (e.g. sbatch, qsub, bash)",
    )
    parser.add_argument(
        "--job-dir",
        default="jobs",
        help="Directory to write per-job scripts",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to write logs when using local bash",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job scripts without running",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for local jobs to finish",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to the script (use -- to separate)",
    )
    args = parser.parse_args()

    lrs = normalize_list(args.lr)
    epochs = normalize_list(args.epochs)
    if not lrs or not epochs:
        raise SystemExit("Both --lr and --epochs must be non-empty.")

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    submit_cmd = shlex.split(args.submit_cmd)
    if not submit_cmd:
        raise SystemExit("--submit-cmd cannot be empty.")

    local = submit_cmd[0] in {"bash", "sh"}
    log_dir = Path(args.log_dir)
    if local:
        log_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(lrs, epochs))
    base = script_path.stem
    directives = extract_scheduler_directives(script_path)

    print(f"Submitting {len(combos)} jobs via: {' '.join(submit_cmd)}")
    processes = []

    for lr, ep in combos:
        run_name = f"lr{safe_token(lr)}_ep{safe_token(ep)}"
        output_dir = f"runs/{run_name}"
        job_name = f"{base}_{run_name}"
        job_path = job_dir / f"{job_name}.sh"

        build_job_script(
            script_path=script_path,
            job_path=job_path,
            lr=lr,
            ep=ep,
            run_name=run_name,
            output_dir=output_dir,
            directives=directives,
            script_args=args.script_args,
        )

        if args.dry_run:
            print(f"[dry-run] {job_path}")
            continue

        if local:
            log_path = log_dir / f"{job_name}.log"
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                submit_cmd + [str(job_path)],
                stdout=log_file,
                stderr=log_file,
            )
            log_file.close()
            processes.append((job_name, proc, log_path))
            print(f"Started {job_name} (pid {proc.pid}) -> {log_path}")
        else:
            subprocess.run(submit_cmd + [str(job_path)], check=True)
            print(f"Submitted {job_name}")

    if local and processes and not args.no_wait:
        print("Waiting for local jobs to finish...")
        for job_name, proc, _ in processes:
            code = proc.wait()
            if code != 0:
                print(f"{job_name} exited with code {code}")


if __name__ == "__main__":
    main()
