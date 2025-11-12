
from pathlib import Path
import numpy as np, json
p = Path("interim_data")

# list produced arrays
arrs = sorted(p.glob("*.npy"))
print("encoded arrays:", [a.name for a in arrs])

# show shapes
for a in arrs:
    x = np.load(a)
    print(f"{a.name:30s}  shape={x.shape}")

# mappings (categorical)
m = p/"mappings.json"
if m.exists():
    mp = json.loads(m.read_text())
    print("mappings:", {k: len(v) for k,v in mp.items()})
else:
    print("no mappings.json (ok if you have no categorical datasets)")