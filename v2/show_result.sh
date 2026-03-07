python3 - <<'PY'
import os, re, ast

root = "/mnt/nushare2/data/baliao/dpc/v2-base-optim_clean-final_fp32/fold0"
rows = []

key = "eval_geo_mean"
hard_key = "eval_hard_geo_mean"

for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        continue
    logp = os.path.join(d, "log.out")
    if not os.path.exists(logp):
        continue

    best = None
    with open(logp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if key not in line:
                continue
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            try:
                rec = ast.literal_eval(m.group(0))
                g = float(str(rec.get(key)))
            except Exception:
                continue
            if best is None or g > best[0]:
                best = (g, rec)

    if best is not None:
        try:
            hard_g = float(str(best[1].get(hard_key)))
        except Exception:
            hard_g = float("nan")
        rows.append((best[0], hard_g, name))

rows.sort(key=lambda x: x[0], reverse=True)

name_w = max(
    len("sub_dir"),
    max((len(name) for _, _, name in rows), default=0),
)

col_best = "best_eval_geo_mean"
col_hard = "eval_hard_geo_mean@best_eval_geo_mean"
col_best_w = len(col_best)
col_hard_w = len(col_hard)

print(
    f"{'rank':>4}  "
    f"{'sub_dir':<{name_w}}  "
    f"{col_best:>{col_best_w}}  "
    f"{col_hard:>{col_hard_w}}"
)
print(
    f"{'-' * 4}  "
    f"{'-' * name_w}  "
    f"{'-' * col_best_w}  "
    f"{'-' * col_hard_w}"
)
for i, (g, hard_g, name) in enumerate(rows, start=1):
    print(
        f"{i:>4}  "
        f"{name:<{name_w}}  "
        f"{g:>{col_best_w}.4f}  "
        f"{hard_g:>{col_hard_w}.4f}"
    )
PY
