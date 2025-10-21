# Evaluation Plan for Robust Quadruped Locomotion Research
**Project:** Proactive Reinforcement Learning for Robust Quadruped Locomotion Under Limb Dropout and Sensor Noise
**Student:** Anand Patel (2561034)
**Date:** October 2025

---

## Research Question & Hypothesis

**Core Research Question:**
> Can a quadruped locomotion policy trained using proactive RL strategies (curriculum-based DR + SR2L + PPO) achieve robustness to actuator failures and sensor noise, and how does performance compare to ablated variants?

**Hypothesis:**
> Domain randomization, smooth regularization, and SR2L contribute **independently** to improved robustness, with their combination yielding the most fault-tolerant policy overall.

---

## 1. Required 4-Way Ablation Study

### Models to Compare:

| Model ID | Name | Training Components | Purpose |
|----------|------|---------------------|---------|
| **M1** | PPO Baseline | PPO only (no robustness) | Baseline performance under ideal conditions |
| **M2** | PPO + SR2L | PPO + Smooth Regularization (λ=0.01) | Isolate SR2L contribution (sensor noise robustness) |
| **M3** | PPO + DR | PPO + Domain Randomization (joint dropout) | Isolate DR contribution (actuator failure robustness) |
| **M4** | Ultimate Combo | PPO + SR2L + DR | Full pipeline (hypothesis: best overall) |

**Rationale:** This directly addresses research proposal requirement for ablation study showing independent contributions.

---

## 2. Core Evaluation Metrics

### 2.1 Primary Performance Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Forward Velocity** | Average m/s over episode | Primary locomotion objective |
| **Distance Traveled** | Net forward displacement (m) | Task completion measure |
| **Success Rate** | % episodes achieving ≥1.5m in 5s | Binary task success criterion |
| **Cumulative Reward** | Total episode reward | Training objective alignment |

### 2.2 Robustness Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Retention Percentage** | (Faulty performance / Baseline performance) × 100% | Relative robustness measure |
| **Recovery Time** | Steps to resume forward motion after fault | Adaptation speed |
| **Failure Rate** | % episodes with collapse/spinning | Catastrophic failure frequency |
| **Performance Degradation** | Velocity drop vs. fault severity | Graceful degradation analysis |

### 2.3 Secondary Analysis Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Gait Stability** | Variance in base orientation | Movement smoothness |
| **Action Smoothness** | Temporal variance in joint commands | SR2L effectiveness measure |
| **Energy Efficiency** | Sum of squared action magnitudes | Practical deployment consideration |

---

## 3. Test Scenarios (Comprehensive Coverage)

### 3.1 Baseline (No Faults)
**Purpose:** Ensure robustness training doesn't sacrifice clean-environment performance

- **Conditions:** No sensor noise, no joint failures
- **Episodes:** 100 rollouts per model
- **Expected:** M4 ≈ M1 performance (no significant trade-off)

---

### 3.2 Sensor Noise Testing (SR2L Specialty)

#### Test Conditions:
| Noise Level | Std. Dev (σ) | Description | Training Exposure |
|-------------|--------------|-------------|-------------------|
| **Clean** | 0.00 | No noise | Baseline condition |
| **1X Training** | 0.01 | Within training distribution | SR2L trained at this level |
| **5X Training** | 0.05 | Moderate extrapolation | 5× training noise |
| **10X Training** | 0.10 | Significant extrapolation | 10× training noise |
| **20X Training** | 0.20 | Extreme stress test | 20× training noise |
| **30X Training** | 0.30 | Near-breakdown condition | 30× training noise |
| **50X Training** | 0.50 | Beyond reasonable limits | 50× training noise |
| **70X Training** | 0.70 | Sensor breakdown simulation | 70× training noise |

**Application:** Gaussian noise added to joint angles and velocities (observations dims 13-28 only, matching SR2L training)

**Episodes:** 100 rollouts per noise level per model

**Expected Results:**
- M2 (PPO+SR2L) should significantly outperform M1 and M3 at high noise levels
- M4 should match or exceed M2 performance

**Key Analysis:**
- Noise vs. velocity retention curves for all 4 models
- Identify noise level where each model drops below 50% retention
- Quantify SR2L's independent contribution

---

### 3.3 Joint Failure Testing (DR Specialty)

#### Test Conditions - Individual Joint Failures:

**Selected Joints** (based on V7.7E empirical findings):
| Joint | Type | V7.7E Retention | Why Test This Joint |
|-------|------|-----------------|---------------------|
| **Hip_1** | Front-left hip | 81.8% | Best hip - easiest case |
| **Hip_4** | Rear-right hip | 43.5% | Rear hip - moderate challenge |
| **Ankle_2** | Front-right ankle | 59.4% | Best ankle - good performance |
| **Ankle_3** | Rear-left ankle | 43.6% | Rear ankle - balanced challenge |

**Failure Implementation:**
- **Delayed locking:** 2-second (120-step) delay before joint locks at 0.0
- **Episode duration:** 30 seconds (1800 frames at 60fps)
- **Justification:** Matches V7.7E DR championship methodology

**Episodes:** 100 rollouts per joint per model

---

#### Test Conditions - Multiple Joint Failures:

| Failure Pattern | Joints Failed | Severity | Purpose |
|-----------------|---------------|----------|---------|
| **Dual Ankle** | Ankle_2 + Ankle_3 | High | Both rear ankles disabled |
| **Dual Hip** | Hip_1 + Hip_4 | Moderate | Diagonal hip failure |
| **Complete Leg** | Hip_1 + Ankle_1 | Extreme | Entire front-left leg |
| **Diagonal** | Hip_1 + Hip_4 | Complex | Cross-body coordination |

**Episodes:** 100 rollouts per pattern per model

**Expected Results:**
- M3 (PPO+DR) should significantly outperform M1 and M2 with joint failures
- M4 should match or exceed M3 performance

---

### 3.4 Combined Stress Testing (Ultimate Challenge)

**Purpose:** Test synergy between SR2L and DR - does M4 handle combined stressors better than M2 or M3 alone?

| Test Scenario | Noise Level | Joint Failure | Purpose |
|---------------|-------------|---------------|---------|
| **Mild Combined** | σ=0.05 | Ankle_2 locked | Realistic combined stress |
| **Moderate Combined** | σ=0.10 | Hip_4 locked | Significant combined stress |
| **Severe Combined** | σ=0.20 | Hip_1 + Ankle_3 | Extreme combined stress |

**Episodes:** 100 rollouts per scenario per model

**Expected Results:**
- M4 should demonstrate **synergistic effect** - outperforming both M2 and M3
- Quantify: M4 retention > max(M2 retention, M3 retention)

---

## 4. Statistical Analysis Plan

### 4.1 Descriptive Statistics
For each model × condition combination:
- Mean and standard deviation
- Median and interquartile range
- 95% confidence intervals
- Distribution plots (violin/box plots)

### 4.2 Comparative Tests

**Numerical Metrics (velocity, distance, reward):**
- Paired t-tests between model pairs
- One-way ANOVA across all 4 models per condition
- Bonferroni correction for multiple comparisons

**Categorical Metrics (success/failure rates):**
- Chi-squared tests for proportion differences
- Fisher's exact test for small sample sizes

**Performance Degradation:**
- Linear regression: Performance ~ Noise/Failure Severity
- Compare slopes between models (interaction analysis)

### 4.3 Hypothesis Testing

**H1:** SR2L contributes independently to sensor noise robustness
- **Test:** M2 > M1 under high noise (p < 0.05)

**H2:** DR contributes independently to actuator failure robustness
- **Test:** M3 > M1 under joint failures (p < 0.05)

**H3:** Combined approach (M4) achieves best overall robustness
- **Test:** M4 ≥ max(M2, M3) across all conditions (p < 0.05)

---

## 5. Visualization Plan

### 5.1 Primary Figures (For Paper)

**Figure 1: Ablation Study Overview**
- 4×4 grid showing all models in all primary conditions
- Bar charts: velocity retention percentage
- Error bars: 95% CI

**Figure 2: Noise Robustness Curves**
- X-axis: Noise level (0.0 → 0.7)
- Y-axis: Velocity retention %
- 4 lines (one per model)
- Shaded regions: ±1 SD

**Figure 3: Joint Failure Performance**
- Grouped bar chart: All 4 joints × 4 models
- Y-axis: Velocity retention %
- Color-coded by model

**Figure 4: Combined Stress Results**
- Heatmap: Noise level × Joint failure severity
- Color: Velocity retention for M4
- Comparison heatmaps for M2, M3 side-by-side

**Figure 5: Independent Contribution Analysis**
- Venn diagram or contribution plot
- Quantify: % improvement from SR2L vs. DR vs. Both

### 5.2 Supplementary Figures

- Distribution plots (violin plots) for each condition
- Recovery time comparisons (box plots)
- Action smoothness analysis (temporal variance)
- Failure mode analysis (why/when models fail)

---

## 6. Additional Analyses (If Time Permits)

### 6.1 Gait Analysis
**Question:** How do robustness strategies affect movement patterns?

- **Metrics:** Step frequency, stride length, base height variance
- **Visualization:** Gait cycle plots, trajectory visualizations
- **Comparison:** Normal gait vs. adapted gait under failures

### 6.2 Generalization Analysis
**Question:** Do models generalize to unseen joint combinations?

- **Test:** Random 3-joint failures (not seen during training)
- **Purpose:** Measure true robustness vs. memorization

### 6.3 Computational Cost Analysis
**Question:** What's the training/inference cost of each approach?

- **Metrics:** Training time, inference latency, memory usage
- **Purpose:** Practical deployment considerations

---

## 7. Research Gaps This Addresses

Based on your proposal (Table 2.1):

| Gap in Literature | How Our Evaluation Addresses It |
|-------------------|----------------------------------|
| **No combined DR+SR2L for internal faults** | Direct comparison shows synergy/interference |
| **Limited actuator + sensor combined testing** | Combined stress scenarios test realistic conditions |
| **Unclear independent contributions** | 4-way ablation quantifies each component |
| **No systematic joint failure analysis** | Empirically-validated joint selection (V7.7E data) |

---

## 8. Timeline & Milestones

| Date Range | Evaluation Tasks | Deliverable |
|------------|------------------|-------------|
| **Oct 6-10** | Run all 4 models through baseline & noise tests | Noise robustness data |
| **Oct 11-15** | Joint failure testing (single & multiple) | Joint failure data |
| **Oct 16-18** | Combined stress testing | Combined stress data |
| **Oct 19-21** | Statistical analysis & hypothesis testing | Results tables |
| **Oct 22-25** | Generate all figures & visualizations | Figures 1-5 |
| **Oct 26-30** | Write Results & Discussion chapters | Draft chapters 4-5 |

---

## 9. Success Criteria

### Minimum Viable Results (Required for Submission):

✅ All 4 models evaluated in all core scenarios
✅ Statistical significance established for key comparisons
✅ Hypothesis about independent contributions tested
✅ Results directly answer research question

### Ideal Results (Strong Paper):

🎯 M4 demonstrates clear synergy (outperforms M2 and M3)
🎯 Quantified contribution percentages (e.g., "SR2L: +20% noise robustness, DR: +30% failure robustness")
🎯 Graceful degradation curves show no performance cliffs
🎯 Real-world applicability demonstrated (energy efficiency maintained)

---

## 10. Questions for Supervisor

### Methodology Questions:
1. **Sample size:** Is 100 rollouts per condition sufficient for statistical power?
2. **Noise levels:** Should we test additional intermediate levels (e.g., 0.15, 0.25)?
3. **Joint failure duration:** Is 30 seconds adequate or should we extend episodes?
4. **Multiple comparisons:** Which correction method do you prefer (Bonferroni vs. Holm-Bonferroni)?

### Scope Questions:
5. **Stretch goals:** Should we prioritize gait analysis or generalization testing if time permits?
6. **Video demonstrations:** How many qualitative videos should we generate for the paper?
7. **Real robot testing:** Is sim-to-real deployment required, or acceptable as future work?

### Analysis Questions:
8. **Synergy quantification:** What's the best way to prove synergy beyond M4 > max(M2, M3)?
9. **Failure mode analysis:** Should we categorize failure types (collapse, spinning, stuck)?
10. **Baseline trade-off:** What's an acceptable clean-environment performance drop for M4 vs M1?

---

## 11. Data Organization

### File Structure:
```
evaluations/
├── experiment_1_baseline/
│   ├── data/
│   │   ├── m1_baseline_results.json
│   │   ├── m2_baseline_results.json
│   │   ├── m3_baseline_results.json
│   │   └── m4_baseline_results.json
│   └── visualizations/
├── experiment_2_noise/
│   ├── data/
│   │   ├── noise_0.00/ ... noise_0.70/
│   └── visualizations/
├── experiment_3_joint_failures/
│   ├── data/
│   │   ├── hip_1/ ... ankle_3/
│   │   ├── multiple_failures/
│   └── visualizations/
├── experiment_4_combined/
│   ├── data/
│   └── visualizations/
└── statistical_analysis/
    ├── hypothesis_tests.ipynb
    ├── results_summary.csv
    └── figures_for_paper/
```

---

## 12. Evaluation Checklist

**Before Starting:**
- [ ] All 4 models trained and validated
- [ ] Evaluation environments configured correctly
- [ ] Data collection scripts tested and verified
- [ ] Storage space allocated (~100GB estimated)

**During Evaluation:**
- [ ] Log all random seeds for reproducibility
- [ ] Monitor for NaN/inf values in metrics
- [ ] Save raw trajectory data for later analysis
- [ ] Generate progress checkpoints every 25 rollouts

**After Evaluation:**
- [ ] Verify data completeness (no missing conditions)
- [ ] Run sanity checks (baseline M1 matches training performance)
- [ ] Back up all raw data before analysis
- [ ] Document any anomalies or unexpected results

---

## Summary

This evaluation plan provides:
✅ **Complete coverage** of research proposal requirements
✅ **Systematic testing** of hypothesis about independent contributions
✅ **Clear metrics** aligned with research question
✅ **Statistical rigor** for publishable results
✅ **Practical scope** achievable in 3-4 weeks
✅ **Flexibility** for additional analyses if time permits

**Next Steps:**
1. Review this plan with supervisor
2. Adjust based on feedback
3. Begin baseline evaluation (Experiment 1)
4. Proceed systematically through Experiments 2-4
