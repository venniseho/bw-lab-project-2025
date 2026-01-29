import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Paths
# -------------------------
CSV_PATH = Path("outputs/metrics/sam_iou.csv")
OUT_PATH = Path("outputs/metrics/iou_scatter_unnormalised_vs_normalised.png")

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)

required = {"iou_orig", "iou_frag", "chance_iou"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# -------------------------
# Normalise IoU
# norm_iou = (iou - chance) / (1 - chance)
# -------------------------
df["norm_iou_orig"] = (df["iou_orig"] - df["chance_iou"]) / (1.0 - df["chance_iou"])
df["norm_iou_frag"] = (df["iou_frag"] - df["chance_iou"]) / (1.0 - df["chance_iou"])

# Optional: clip for visualization (do NOT clip in CSV)
df_plot = df.copy()
df_plot["norm_iou_orig"] = df_plot["norm_iou_orig"].clip(-0.2, 1.05)
df_plot["norm_iou_frag"] = df_plot["norm_iou_frag"].clip(-0.2, 1.05)

# -------------------------
# Plot
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False)

# ---- Left: Unnormalised IoU ----
ax = axes[0]
ax.scatter(df["iou_orig"], df["iou_frag"], alpha=0.7)
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("IoU (original)")
ax.set_ylabel("IoU (fragmented)")
ax.set_title("Unnormalised IoU")

# ---- Right: Normalised IoU ----
ax = axes[1]
ax.scatter(df_plot["norm_iou_orig"], df_plot["norm_iou_frag"], alpha=0.7)
ax.plot([-0.2, 1.0], [-0.2, 1.0], "k--", linewidth=1)
ax.axhline(0, color="gray", linestyle=":", linewidth=1)
ax.axvline(0, color="gray", linestyle=":", linewidth=1)
ax.set_xlim(-0.2, 1.0)
ax.set_ylim(-0.2, 1.0)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Normalised IoU (original)")
ax.set_ylabel("Normalised IoU (fragmented)")
ax.set_title("Normalised IoU\n(0 = chance level)")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved scatter plot -> {OUT_PATH}")
