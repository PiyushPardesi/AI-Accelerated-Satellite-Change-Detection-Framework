import pandas as pd
import matplotlib.pyplot as plt

# Load the new 637-image data
df = pd.read_csv("times.csv")

# Calculate Speedup
seq_time = df[df["Method"]=="Sequential"]["Time_sec"].values[0]
df["Speedup"] = seq_time / df["Time_sec"]

# Set professional color palette
colors = ['#4C72B0', '#55A868', '#C44E52']

# ---- Bar chart: Execution Time ----
plt.figure(figsize=(8, 5))
bars = plt.bar(df["Method"], df["Time_sec"], color=colors)
plt.ylabel("Total Execution Time (seconds)", fontsize=12)
plt.title("Execution Latency: 637 Image Batch (LEVIR-CD)", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("execution_time_637.png", dpi=300)

# ---- Speedup chart ----
plt.figure(figsize=(8, 5))
bars = plt.bar(df["Method"], df["Speedup"], color=colors)
plt.ylabel("Speedup Factor (x)", fontsize=12)
plt.title("Throughput Speedup relative to Sequential Baseline", fontsize=14, fontweight='bold')
plt.axhline(y=1, color='black', linestyle='-', linewidth=0.8) # Baseline at 1x
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}x', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("speedup_637.png", dpi=300)

print("Graphs generated successfully with new 637-image data.")
print(df)