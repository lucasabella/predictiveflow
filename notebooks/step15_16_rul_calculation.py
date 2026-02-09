"""
Step 15 & 16: Calculate RUL and Cap at 125

Step 15: For each row, RUL = max_cycle_for_engine - current_cycle
Step 16: Cap RUL at 125 (early life = healthy, no meaningful difference)
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

# ============================================================================
# Step 15: Calculate RUL for every row
# Logic: RUL = max_cycle_for_engine - current_cycle
# ============================================================================

print("\n" + "="*60)
print("STEP 15: CALCULATING RUL")
print("="*60)

# Get the max cycle for each engine (this is when it failed)
max_cycles = train_df.groupby("engine_id")["cycle"].max().rename("max_cycle")
print(f"\nMax cycles per engine (first 10):")
print(max_cycles.head(10))

# Merge max_cycle back to the dataframe
train_df = train_df.merge(max_cycles, on="engine_id")

# Calculate RUL: remaining cycles until failure
train_df["rul"] = train_df["max_cycle"] - train_df["cycle"]

# Verify: first and last rows for a few engines
print("\nVerification - Engine 1:")
eng1 = train_df[train_df["engine_id"] == 1][["engine_id", "cycle", "max_cycle", "rul"]]
print(f"  First row (cycle 1): RUL = {eng1.iloc[0]['rul']} (should be {eng1.iloc[0]['max_cycle'] - 1})")
print(f"  Last row (cycle {eng1.iloc[-1]['cycle']}): RUL = {eng1.iloc[-1]['rul']} (should be 0)")

print("\nRUL statistics (before capping):")
print(train_df["rul"].describe())

# ============================================================================
# Step 16: Cap RUL at 125
# Why? Early life = healthy, no meaningful difference between 300 and 250 cycles left
# ============================================================================

print("\n" + "="*60)
print("STEP 16: CAPPING RUL AT 125")
print("="*60)

RUL_CAP = 125

train_df["rul_capped"] = train_df["rul"].clip(upper=RUL_CAP)

print(f"\nRUL capping: any RUL > {RUL_CAP} becomes {RUL_CAP}")
print(f"Rows affected: {(train_df['rul'] > RUL_CAP).sum()} out of {len(train_df)} ({100*(train_df['rul'] > RUL_CAP).sum()/len(train_df):.1f}%)")

print("\nRUL statistics (after capping):")
print(train_df["rul_capped"].describe())

# ============================================================================
# Visualize the RUL distributions
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw RUL distribution
axes[0].hist(train_df["rul"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
axes[0].set_xlabel("RUL (cycles)")
axes[0].set_ylabel("Count")
axes[0].set_title("Raw RUL Distribution")
axes[0].axvline(x=RUL_CAP, color="red", linestyle="--", linewidth=2, label=f"Cap at {RUL_CAP}")
axes[0].legend()

# Capped RUL distribution
axes[1].hist(train_df["rul_capped"], bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_xlabel("RUL (cycles, capped)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Capped RUL Distribution (max = {RUL_CAP})")

plt.tight_layout()
plt.savefig("../data/processed/step15_16_rul_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# Visualize RUL over engine lifetime for a few engines
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sample_engines = [1, 20, 50, 80, 100]
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_engines)))

# Raw RUL
ax = axes[0]
for i, eng_id in enumerate(sample_engines):
    eng_data = train_df[train_df["engine_id"] == eng_id]
    ax.plot(eng_data["cycle"], eng_data["rul"], color=colors[i], linewidth=2, 
            label=f"Engine {eng_id} (max={eng_data['max_cycle'].iloc[0]})")
ax.set_xlabel("Cycle")
ax.set_ylabel("RUL")
ax.set_title("Raw RUL Over Engine Lifetime")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# Capped RUL
ax = axes[1]
for i, eng_id in enumerate(sample_engines):
    eng_data = train_df[train_df["engine_id"] == eng_id]
    ax.plot(eng_data["cycle"], eng_data["rul_capped"], color=colors[i], linewidth=2,
            label=f"Engine {eng_id}")
ax.axhline(y=RUL_CAP, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"Cap = {RUL_CAP}")
ax.set_xlabel("Cycle")
ax.set_ylabel("RUL (capped)")
ax.set_title(f"Capped RUL Over Engine Lifetime (cap = {RUL_CAP})")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../data/processed/step15_16_rul_over_lifetime.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
STEP 15 - RUL Calculation:
- RUL = max_cycle - current_cycle for each engine
- Last cycle has RUL = 0 (engine failed)
- First cycle has RUL = max_cycle - 1

STEP 16 - RUL Capping at {RUL_CAP}:
- Rows with RUL > {RUL_CAP} are capped to {RUL_CAP}
- {(train_df['rul'] > RUL_CAP).sum()} rows affected ({100*(train_df['rul'] > RUL_CAP).sum()/len(train_df):.1f}%)
- This creates a piecewise linear target

WHY cap at 125?
1. Early life → engine is healthy, sensors stable
2. No meaningful difference between "300 cycles left" vs "250 cycles left"  
3. Model should focus on the DEGRADATION phase (RUL < 125)
4. Reduces extreme values, making regression easier
5. Industry standard for C-MAPSS dataset

Final columns added:
- 'rul': raw remaining useful life
- 'rul_capped': RUL capped at {RUL_CAP}
""")

# Save the processed dataframe
train_df.to_csv("../data/processed/train_FD001_with_rul.csv", index=False)
print(f"\nSaved processed data to: data/processed/train_FD001_with_rul.csv")
print(f"Columns: {list(train_df.columns)}")
