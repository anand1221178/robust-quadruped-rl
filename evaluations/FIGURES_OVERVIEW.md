# Evaluation Figures Overview

## What Gets Plotted?

The evaluation suite generates **5 publication-quality figures** from your experiment results.

---

## Figure 1: Baseline Performance Comparison
**File:** `figures/figure_1_baseline_comparison.png`
**Size:** 18" × 6" (3 subplots side-by-side)

### What it shows:
- **Left panel:** Distance traveled (m) - bar chart with error bars
- **Middle panel:** Success rate (%) - bar chart with 50% threshold line
- **Right panel:** Failure rate (%) - bar chart showing robot collapses

### Models compared:
- PPO Baseline (red)
- PPO + SR2L (teal)
- PPO + DR (light green)
- Ultimate Combo (yellow)

### Purpose:
Shows baseline performance under ideal conditions (no faults). Tests whether robustness training sacrifices clean-environment performance.

### Expected insight:
- M1 (baseline) likely highest distance
- Validates no catastrophic trade-offs
- Establishes reference performance levels

---

## Figure 2: Sensor Noise Robustness Curves
**File:** `figures/figure_2_sensor_noise_robustness.png`
**Size:** 16" × 12" (4 subplots in 2×2 grid)

### What it shows:

#### Top-left: Distance vs Noise
- Line plot with 8 points (noise levels 0.0 → 0.7)
- Shaded confidence bands (±1 std)
- Vertical line at σ=0.01 (training noise level)
- Shows how distance degrades with increasing noise

#### Top-right: Success Rate vs Noise
- Line plot showing % episodes achieving ≥1.5m
- Horizontal line at 50% threshold
- Shows catastrophic failure points

#### Bottom-left: Failure Rate vs Noise
- Line plot showing % episodes with robot collapse
- Shows stability degradation

#### Bottom-right: Retention Heatmap
- Color-coded matrix: models × noise levels
- Shows retention % (distance relative to baseline)
- Green = good retention, Red = poor retention

### Purpose:
**Tests Hypothesis H1:** SR2L contributes independently to sensor noise robustness

### Expected insight:
- **M2 (SR2L) should dominate** - flat retention even at high noise
- M1 (baseline) and M3 (DR) should degrade significantly
- M4 should match M2 (inherits SR2L robustness)
- Quantifies SR2L's independent contribution

---

## Figure 3: Joint Failure Robustness Heatmap
**File:** `figures/figure_3_joint_failure_robustness.png`
**Size:** 16" × 12" (4 subplots in 2×2 grid)

### What it shows:

#### Top-left: Distance Heatmap
- Color-coded matrix: models × 8 joints (hip_1, ankle_1, ..., hip_4, ankle_4)
- Shows absolute distance (m) with each joint failure
- Warmer colors = better performance

#### Top-right: Success Rate Heatmap
- Color-coded matrix: models × 8 joints
- Shows % episodes achieving ≥1.5m
- Green (100%) to Red (0%)

#### Bottom-left: Average Performance
- Bar chart showing average distance across all joint failures
- Shows which model handles failures best overall

#### Bottom-right: Per-Joint Comparison
- Grouped bar chart: 8 joints with 4 bars each (one per model)
- Shows which joints are hardest (e.g., ankle_4)
- Direct model comparison per joint

### Purpose:
**Tests Hypothesis H2:** DR contributes independently to joint failure robustness

### Expected insight:
- **M3 (DR) should dominate** - trained specifically for joint failures
- M1 (baseline) and M2 (SR2L) should struggle
- M4 should match M3 (inherits DR robustness)
- Identifies hardest joints (likely ankle_4)

---

## Figure 4: Combined Stress Results
**File:** `figures/figure_4_combined_stress.png`
**Size:** 16" × 12" (4 subplots in 2×2 grid)

### What it shows:

#### Top-left: Distance Under Combined Stress
- Grouped bar chart: 6 scenarios with 4 bars each
- Scenarios: Mild, Moderate, Challenging, Severe, Extreme Dual, Ultimate
- Shows performance under simultaneous noise + joint failures

#### Top-right: Success Rate Under Combined Stress
- Grouped bar chart: same 6 scenarios
- Shows % episodes achieving ≥1.5m
- 50% threshold line

#### Bottom-left: Average Performance
- Bar chart: average distance across all combined scenarios
- **Highlights best performer** (gold border)
- Shows overall combined stress champion

#### Bottom-right: Synergy Analysis
- Bar chart comparing:
  - M2 (SR2L) average
  - M3 (DR) average
  - Max(M2, M3) - best specialist
  - M4 (Combo) - combined approach
- **Visual synergy indicator:**
  - ✓ Green "Synergy +X%" if M4 > max(M2, M3)
  - ✗ Red "No Synergy" if M4 ≤ max(M2, M3)

### Purpose:
**Tests Hypothesis H3:** Combined approach achieves best overall robustness via synergy

### Expected insight:
- **M4 should outperform both M2 and M3** under combined stress
- Neither M2 nor M3 alone sufficient
- Demonstrates value of combined pipeline
- Quantifies synergy magnitude

---

## Figure 5: Comprehensive Summary Table
**File:** `figures/figure_5_comprehensive_summary.png`
**Size:** 16" × 10"

### What it shows:

**Table with columns:**
1. Model name
2. Baseline distance (from Exp 1)
3. Noise robustness (distance at σ=0.1 from Exp 2)
4. Joint failure robustness (average across all joints from Exp 3)
5. Combined stress performance (average across all scenarios from Exp 4)
6. Overall rank (TBD - based on weighted scoring)

**Styling:**
- Color-coded model rows (matching model colors)
- Header row highlighted
- Easy at-a-glance comparison

### Purpose:
Executive summary figure showing all key metrics in one place

### Expected insight:
- Quick comparison across all test conditions
- Identifies overall best performer
- Shows specialization patterns (M2 best at noise, M3 best at failures, M4 best combined)

---

## How to Generate Figures

### After running all experiments:
```bash
cd evaluations
python analyze_and_visualize.py
```

### What it does:
1. Automatically finds most recent result files from each experiment
2. Loads JSON data for all 4 models
3. Generates all 5 figures
4. Saves to `evaluations/figures/` directory

### Requirements:
- All 4 experiments must be run first (or at least some of them)
- JSON result files in `experiment_*/data/` directories
- Python packages: `matplotlib`, `seaborn`, `numpy`, `pandas`, `scipy`

---

## Figure Quality

**All figures are publication-ready:**
- ✅ 300 DPI resolution
- ✅ Professional styling (seaborn whitegrid)
- ✅ Clear labels and titles
- ✅ Consistent color scheme
- ✅ Legends and annotations
- ✅ Grid lines for readability
- ✅ Error bars where appropriate
- ✅ Statistical indicators

---

## Customization

Want to modify figures? Edit `analyze_and_visualize.py`:

**Change colors:**
```python
self.model_colors = {
    'M1_baseline': '#FF6B6B',  # Your color here
    ...
}
```

**Change figure size:**
```python
fig, ax = plt.subplots(figsize=(width, height))
```

**Add more metrics:**
```python
# In each plot_X function, add your own subplots
```

**Change statistical tests:**
```python
from scipy import stats
# Add t-tests, ANOVA, etc.
```

---

## Expected Results (Based on Hypotheses)

### Baseline (Figure 1):
- M1 likely highest (no robustness overhead)
- All models within 10-20% of each other

### Sensor Noise (Figure 2):
- **M2 (SR2L) should show flat retention curve**
- M1, M3 should show steep degradation
- M4 should match M2 performance

### Joint Failures (Figure 3):
- **M3 (DR) should show highest average distance**
- M1, M2 should show significant degradation
- M4 should match M3 performance
- Ankle_4 hardest for all models

### Combined Stress (Figure 4):
- **M4 should outperform max(M2, M3)**
- Synergy indicator should be GREEN ✓
- Demonstrates value of combined pipeline

### Summary (Figure 5):
- M1: Best baseline, poor robustness
- M2: Moderate baseline, excellent noise robustness
- M3: Moderate baseline, excellent failure robustness
- M4: Moderate baseline, **best combined robustness**

---

## File Organization

```
evaluations/
├── experiment_1_baseline/
│   └── data/
│       └── baseline_results_20251013_093045.json
├── experiment_2_sensor_noise/
│   └── data/
│       └── sensor_noise_results_20251013_102134.json
├── experiment_3_joint_failures/
│   └── data/
│       └── joint_failure_results_20251013_153442.json
├── experiment_4_combined_stress/
│   └── data/
│       └── combined_stress_results_20251013_193721.json
├── figures/  ← Created by analyze_and_visualize.py
│   ├── figure_1_baseline_comparison.png
│   ├── figure_2_sensor_noise_robustness.png
│   ├── figure_3_joint_failure_robustness.png
│   ├── figure_4_combined_stress.png
│   └── figure_5_comprehensive_summary.png
├── analyze_and_visualize.py  ← Run this to generate figures
└── FIGURES_OVERVIEW.md  ← This file
```

---

## Next Steps After Figure Generation

1. **Review figures:** Check if results match expected hypotheses
2. **Statistical tests:** Run t-tests, ANOVA (see EVALUATION_PLAN.md)
3. **Write Results chapter:**
   - Figure 1 → Section 4.1 (Baseline Performance)
   - Figure 2 → Section 4.2 (Sensor Noise Robustness)
   - Figure 3 → Section 4.3 (Joint Failure Robustness)
   - Figure 4 → Section 4.4 (Combined Stress & Synergy)
   - Figure 5 → Section 4.5 (Summary & Discussion)

4. **Update research template:** Fill in RESEARCH_ABLATION_STUDY_TEMPLATE.md with actual values

---

*Ready to visualize? Run:* `python analyze_and_visualize.py`
