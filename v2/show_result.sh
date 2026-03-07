python3 - <<'PY'
import os, re, ast

root = "/mnt/nushare2/data/baliao/dpc/v2-base-optim_clean-final_fp32/fold0"
rows_best_geo = []
rows_best_hard = []

key = "eval_geo_mean"
hard_key = "eval_hard_geo_mean"

for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        continue
    logp = os.path.join(d, "log.out")
    if not os.path.exists(logp):
        continue

    best_geo = None
    best_hard = None
    with open(logp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if key not in line and hard_key not in line:
                continue
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            try:
                rec = ast.literal_eval(m.group(0))
                g = float(str(rec.get(key)))
                hard_g = float(str(rec.get(hard_key)))
            except Exception:
                continue
            if best_geo is None or g > best_geo[0]:
                best_geo = (g, hard_g)
            if best_hard is None or hard_g > best_hard[1]:
                best_hard = (g, hard_g)

    if best_geo is not None:
        rows_best_geo.append((best_geo[0], best_geo[1], name))
    if best_hard is not None:
        rows_best_hard.append((best_hard[0], best_hard[1], name))

rows_best_geo.sort(key=lambda x: x[0], reverse=True)
rows_best_hard.sort(key=lambda x: x[1], reverse=True)

def print_table(title, rows, col_1, col_2):
    name_w = max(
        len("sub_dir"),
        max((len(name) for _, _, name in rows), default=0),
    )
    col_1_w = len(col_1)
    col_2_w = len(col_2)

    print(title)
    print(
        f"{'rank':>4}  "
        f"{'sub_dir':<{name_w}}  "
        f"{col_1:>{col_1_w}}  "
        f"{col_2:>{col_2_w}}"
    )
    print(
        f"{'-' * 4}  "
        f"{'-' * name_w}  "
        f"{'-' * col_1_w}  "
        f"{'-' * col_2_w}"
    )
    for i, (v1, v2, name) in enumerate(rows, start=1):
        print(
            f"{i:>4}  "
            f"{name:<{name_w}}  "
            f"{v1:>{col_1_w}.4f}  "
            f"{v2:>{col_2_w}.4f}"
        )

print_table(
    "Table 1: best eval_geo_mean",
    rows_best_geo,
    "best_eval_geo_mean",
    "eval_hard_geo_mean@best_eval_geo_mean",
)

print()
print_table(
    "Table 2: best eval_hard_geo_mean",
    rows_best_hard,
    "eval_geo_mean@best_eval_hard_geo_mean",
    "best_eval_hard_geo_mean",
)
PY
