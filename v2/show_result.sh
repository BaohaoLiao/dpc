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

print("sub_dir\tbest_eval_geo_mean\teval_hard_geo_mean_at_best_eval_geo_mean")
for g, hard_g, name in rows:
    print(f"{name}\t{g:.4f}\t{hard_g:.4f}")
PY
