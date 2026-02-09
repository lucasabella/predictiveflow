"""
Step 13: Degradation Patterns Across Multiple Engines
- Pick 3-4 sensors that clearly show degradation
- Plot these for 5 different engines on the same chart
- Observe: all engines show similar degradation curves at different speeds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load the raw FD001 training data
columns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + \
          [f"sensor_{i}" for i in range(1, 22)]

train_df = pd.read_csv("../data/raw/train_FD001.txt", sep=r"\s+", header=None, names=columns)

# ============================================================================
# Select sensors that show clear degradation (based on Step 12 analysis)
# These sensors showed clear upward or downward trends as engines degrade:
# ============================================================================

degradation_sensors = ["sensor_2", "sensor_4", "sensor_11", "sensor_21"]

# Select 5 engines with different lifetimes for comparison
selected_engines = [1, 20, 50, 80, 100]
engine_lives = train_df.groupby("engine_id")["cycle"].max()
print(f"\nSelected engines and their lifetimes:")
for e in selected_engines:
    print(f"  Engine {e}: {engine_lives[e]} cycles")

# ============================================================================
# Create the multi-engine comparison plots
# ============================================================================

colors = plt.cm.viridis(np.linspace(0, 1, len(selected_engines)))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Degradation Patterns: Same Sensors, 5 Different Engines", fontsize=14, y=1.02)

for idx, sensor in enumerate(degradation_sensors):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    for i, eng_id in enumerate(selected_engines):
        eng_data = train_df[train_df["engine_id"] == eng_id]
        lifetime = eng_data["cycle"].max()
        ax.plot(eng_data["cycle"], eng_data[sensor], 
                color=colors[i], linewidth=1.5, alpha=0.8,
                label=f"Engine {eng_id} ({lifetime} cycles)")
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel(sensor)
    ax.set_title(sensor)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../data/processed/step13_degradation_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# Normalized view: all engines on same 0-100% lifetime scale
# This better shows the pattern similarity
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Degradation Patterns (Normalized Lifetime: 0% = start, 100% = failure)", fontsize=14, y=1.02)

for idx, sensor in enumerate(degradation_sensors):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    for i, eng_id in enumerate(selected_engines):
        eng_data = train_df[train_df["engine_id"] == eng_id].copy()
        lifetime = eng_data["cycle"].max()
        # Normalize cycle to percentage of lifetime
        eng_data["lifecycle_pct"] = eng_data["cycle"] / lifetime * 100
        
        ax.plot(eng_data["lifecycle_pct"], eng_data[sensor], 
                color=colors[i], linewidth=1.5, alpha=0.8,
                label=f"Engine {eng_id} ({lifetime} cycles)")
    
    ax.set_xlabel("Lifecycle (%)")
    ax.set_ylabel(sensor)
    ax.set_title(f"{sensor} — Normalized Lifecycle")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../data/processed/step13_degradation_normalized.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*60)
print("STEP 13 SUMMARY")
print("="*60)
print(f"""
Selected degradation sensors: {degradation_sensors}

Key observations:
1. All engines show SIMILAR degradation patterns (curves have same shape)
2. Engines degrade at DIFFERENT SPEEDS (some fail in 128 cycles, others in 362)
3. The normalized view shows degradation curves overlap when scaled

This confirms:
- These sensors capture real degradation information
- The degradation process is consistent across engines
- We need to predict WHEN failure occurs, not IF (all engines eventually fail)
- Models need to learn the "shape" of degradation, not just absolute values

Sensors chosen and why:
- sensor_2:  Clear upward trend as engine degrades
- sensor_4:  Shows gradual increase toward failure
- sensor_11: Trends upward with degradation
- sensor_21: Clear trend pattern toward end of life
""")
