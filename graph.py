import matplotlib.pyplot as plt
import pandas as pd

data = {
    "batch": [1, 2, 3, 4, 5],
    "regular_ids_accuracy": [0.60, 0.65, 0.70, 0.72, 0.68],
    "custom_ids_accuracy": [0.85, 0.88, 0.90, 0.91, 0.93]
}

df = pd.DataFrame(data)

plt.style.use("dark_background")
plt.figure(figsize=(10, 6))

plt.plot(df["batch"], df["regular_ids_accuracy"], marker="o", color="#ff69b4", label="Regular IDS", linewidth=2)
plt.plot(df["batch"], df["custom_ids_accuracy"], marker="s", color="#ffc0cb", label="Our IDS model", linewidth=2)

plt.title("IDS Accuracy Comparison Over Batches", fontsize=16, color="white")
plt.xlabel("Batch Number", fontsize=12, color="white")
plt.ylabel("Accuracy", fontsize=12, color="white")
plt.xticks(color='white')
plt.yticks(color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.savefig("ids_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()