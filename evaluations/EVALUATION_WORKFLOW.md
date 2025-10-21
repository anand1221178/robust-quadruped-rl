# Complete Evaluation Workflow

## 📋 Step-by-Step Guide

### Step 1: Run All Experiments (~14-15 hours)
```bash
cd evaluations
python run_all_experiments.py
```

**What happens:**
- 4 experiments run sequentially
- 9,200 total episodes
- Results saved to JSON files
- Progress shown with tqdm bars

**Output files:**
- `experiment_1_baseline/data/baseline_results_TIMESTAMP.json`
- `experiment_2_sensor_noise/data/sensor_noise_results_TIMESTAMP.json`
- `experiment_3_joint_failures/data/joint_failure_results_TIMESTAMP.json`
- `experiment_4_combined_stress/data/combined_stress_results_TIMESTAMP.json`

---

### Step 2: Generate Figures (~1 minute)
```bash
cd evaluations
python analyze_and_visualize.py
```

**What happens:**
- Loads most recent results from each experiment
- Generates 5 publication-quality figures
- Saves to `evaluations/figures/` directory

**Output files:**
- `figures/figure_1_baseline_comparison.png`
- `figures/figure_2_sensor_noise_robustness.png`
- `figures/figure_3_joint_failure_robustness.png`
- `figures/figure_4_combined_stress.png`
- `figures/figure_5_comprehensive_summary.png`

---

### Step 3: Statistical Analysis
```bash
cd evaluations
# Create and run statistical analysis script
python statistical_tests.py  # To be created
```

**What to test:**
- **H1:** M2 (SR2L) > M1 (Baseline) under high noise (paired t-test)
- **H2:** M3 (DR) > M1 (Baseline) with joint failures (paired t-test)
- **H3:** M4 (Combo) > max(M2, M3) under combined stress (paired t-test)

---

### Step 4: Write Results Chapter

Use generated figures in your thesis:

**Section 4.1: Baseline Performance**
- Reference Figure 1
- Report mean distances, success rates
- Compare all 4 models

**Section 4.2: Sensor Noise Robustness**
- Reference Figure 2
- Test H1: SR2L contribution
- Report retention curves

**Section 4.3: Joint Failure Robustness**
- Reference Figure 3
- Test H2: DR contribution
- Report per-joint performance

**Section 4.4: Combined Stress & Synergy**
- Reference Figure 4
- Test H3: Synergy detection
- Report M4 vs max(M2, M3)

**Section 4.5: Discussion**
- Reference Figure 5
- Overall comparison
- Practical implications

---

## 📊 What Gets Plotted (Quick Reference)

### Figure 1: Baseline Comparison (3 panels)
1. Distance bars with error bars
2. Success rate bars with 50% line
3. Failure rate bars

### Figure 2: Sensor Noise (4 panels, 2×2)
1. Distance vs noise curves with confidence bands
2. Success rate vs noise curves
3. Failure rate vs noise curves
4. Retention heatmap (models × noise levels)

### Figure 3: Joint Failures (4 panels, 2×2)
1. Distance heatmap (models × 8 joints)
2. Success rate heatmap (models × 8 joints)
3. Average performance bar chart
4. Per-joint grouped bars

### Figure 4: Combined Stress (4 panels, 2×2)
1. Distance grouped bars (6 scenarios)
2. Success rate grouped bars (6 scenarios)
3. Average performance with best highlighted
4. **Synergy analysis** comparing M4 vs max(M2, M3)

### Figure 5: Summary Table
- Single comprehensive table
- 5 columns: Model, Baseline, Noise, Failures, Combined, Rank
- Color-coded by model

---

## ⏱️ Time Budget

| Step | Duration | Can Run In Background? |
|------|----------|----------------------|
| Experiments | 14-15 hrs | ✅ Yes (overnight) |
| Figures | 1 min | ❌ No (very fast) |
| Statistics | 10-20 min | ❌ No (interactive) |
| Writing | 1-2 days | ❌ No (requires analysis) |

**Total:** ~3-4 days from start to complete Results chapter

---

## 🎯 Expected Results

Based on your research hypothesis:

### Baseline (Figure 1)
- M1 (Baseline): **HIGHEST** distance (~10-11m)
- M2 (SR2L): Moderate (~9-10m, -10%)
- M3 (DR): Moderate (~8-9m, -15%)
- M4 (Combo): Moderate (~9-10m, -10%)

**Interpretation:** Robustness training has small performance cost

### Sensor Noise (Figure 2)
- M1 (Baseline): **STEEP DEGRADATION** at high noise
- **M2 (SR2L): FLAT RETENTION** even at σ=0.7
- M3 (DR): Steep degradation (not trained for noise)
- **M4 (Combo): FLAT RETENTION** (inherits from SR2L)

**Interpretation:** SR2L provides independent sensor noise robustness

### Joint Failures (Figure 3)
- M1 (Baseline): **STEEP DEGRADATION** with failures
- M2 (SR2L): Steep degradation (not trained for failures)
- **M3 (DR): BEST AVERAGE** across all joints
- **M4 (Combo): MATCHES M3** (inherits from DR)

**Interpretation:** DR provides independent joint failure robustness

### Combined Stress (Figure 4)
- M1 (Baseline): Poor performance
- M2 (SR2L): Handles noise but not failures
- M3 (DR): Handles failures but not noise
- **M4 (Combo): OUTPERFORMS MAX(M2, M3)** ✓

**Interpretation:** Synergy confirmed - combined approach superior

---

## 📝 Checklist

**Before Running:**
- [ ] All 4 models exist in `done/` directory
- [ ] Sufficient disk space (~5GB for results)
- [ ] Python environment with required packages
- [ ] 14-15 hours available for full run

**After Experiments:**
- [ ] All 4 JSON result files created
- [ ] No errors in experiment logs
- [ ] Results look reasonable (no NaN, no negative values)

**After Visualization:**
- [ ] All 5 figures generated
- [ ] Figures display correctly (no blank plots)
- [ ] Results match expected hypotheses

**For Paper:**
- [ ] Figures included in thesis
- [ ] Statistical tests performed
- [ ] Results chapter written
- [ ] Discussion interprets findings

---

## 🆘 Troubleshooting

### "No results found" when running analyze_and_visualize.py
**Fix:** Run experiments first! The script needs JSON files to load.

### Figures look wrong or have blank subplots
**Fix:** Check experiment completed successfully. Look for errors in JSON files.

### Want different plot styles
**Fix:** Edit `analyze_and_visualize.py` - change colors, sizes, layouts

### Need additional statistical tests
**Fix:** Use result JSONs to load data and run custom tests with scipy.stats

---

## 📚 Documentation Files

- **README.md** - Complete evaluation suite documentation
- **QUICKSTART.md** - Quick start guide with examples
- **FIGURES_OVERVIEW.md** - Detailed description of all figures
- **EVALUATION_WORKFLOW.md** - This file (step-by-step guide)
- **EVALUATION_PLAN.md** - Research methodology and hypothesis testing

---

*Ready to start? Run:* `cd evaluations && python run_all_experiments.py`
