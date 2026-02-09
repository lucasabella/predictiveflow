"""
Step 11 & 12: Sensor Analysis for Engine 1
- Plot all 21 sensors over engine lifetime
- Identify sensors that barely change (useless for RUL prediction)
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

print(f"Training data shape: {train_df.shape}")
print(f"Number of engines: {train_df['engine_id'].nunique()}")

# Filter for engine 1
engine_1 = train_df[train_df["engine_id"] == 1].copy()
print(f"\nEngine 1 data: {engine_1.shape}")
print(f"Engine 1 lifetime: {engine_1['cycle'].max()} cycles")

# ============================================================================
# Step 11: Plot all 21 sensors for Engine 1
# ============================================================================

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

fig, axes = plt.subplots(7, 3, figsize=(15, 20))
fig.suptitle("Engine 1 — All 21 Sensors Over Lifetime", fontsize=14, y=1.02)

for idx, sensor in enumerate(sensor_cols):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    ax.plot(engine_1["cycle"], engine_1[sensor], color="steelblue", linewidth=1)
    ax.set_title(sensor, fontsize=10)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../data/processed/engine1_all_sensors.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# Step 12: Identify useless sensors (near-constant values)
# ============================================================================

print("\n" + "="*60)
print("STEP 12: IDENTIFYING USELESS SENSORS")
print("="*60)

# Calculate statistics for each sensor for Engine 1
sensor_stats = pd.DataFrame({
    "sensor": sensor_cols,
    "mean": [engine_1[s].mean() for s in sensor_cols],
    "std": [engine_1[s].std() for s in sensor_cols],
    "range": [engine_1[s].max() - engine_1[s].min() for s in sensor_cols],
    "coef_var": [engine_1[s].std() / engine_1[s].mean() if engine_1[s].mean() != 0 else 0 for s in sensor_cols]
})

# Also calculate for ALL engines (more representative)
all_stats = pd.DataFrame({
    "sensor": sensor_cols,
    "mean_all": [train_df[s].mean() for s in sensor_cols],
    "std_all": [train_df[s].std() for s in sensor_cols],
    "range_all": [train_df[s].max() - train_df[s].min() for s in sensor_cols],
    "coef_var_all": [train_df[s].std() / train_df[s].mean() if train_df[s].mean() != 0 else 0 for s in sensor_cols]
})

sensor_stats = sensor_stats.merge(all_stats, on="sensor")

print("\nSensor Statistics (sorted by coefficient of variation across all engines):")
print(sensor_stats.sort_values("coef_var_all")[["sensor", "std", "range", "std_all", "range_all", "coef_var_all"]])

# Identify flat sensors: very low standard deviation or coefficient of variation
# Using threshold: coefficient of variation < 0.01 (1%)
FLAT_THRESHOLD = 0.01

flat_sensors = sensor_stats[sensor_stats["coef_var_all"] < FLAT_THRESHOLD]["sensor"].tolist()

print(f"\n{'='*60}")
print("FLAT (USELESS) SENSORS (coefficient of variation < 1%):")
print("="*60)
print(flat_sensors)

# Also show sensors with very low std (alternative check)
print("\n\nSensors sorted by standard deviation (all engines):")
for _, row in sensor_stats.sort_values("std_all").iterrows():
    print(f"  {row['sensor']:12s} → std={row['std_all']:.4f}, range={row['range_all']:.4f}, CV={row['coef_var_all']:.4f}")

# Create a bar chart of coefficient of variation
fig, ax = plt.subplots(figsize=(14, 6))
colors = ["red" if cv < FLAT_THRESHOLD else "steelblue" for cv in sensor_stats.sort_values("coef_var_all")["coef_var_all"]]
sensor_stats_sorted = sensor_stats.sort_values("coef_var_all")
ax.bar(range(21), sensor_stats_sorted["coef_var_all"], color=colors, edgecolor="black")
ax.set_xticks(range(21))
ax.set_xticklabels(sensor_stats_sorted["sensor"], rotation=45, ha="right")
ax.set_ylabel("Coefficient of Variation")
ax.set_title("Sensor Variability — Red = Near-Constant (Useless)")
ax.axhline(y=FLAT_THRESHOLD, color="red", linestyle="--", label=f"Threshold = {FLAT_THRESHOLD}")
ax.legend()
plt.tight_layout()
plt.savefig("../data/processed/sensor_variability.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Based on the analysis, the following sensors show minimal variation 
and provide no useful degradation information:

USELESS SENSORS: {flat_sensors}

These sensors should be DROPPED during preprocessing because:
- They have near-constant values across all cycles and engines
- They do not capture any degradation pattern
- Including them adds noise and dimensionality without information

Expected useless sensors per documentation: [sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19]
""")
