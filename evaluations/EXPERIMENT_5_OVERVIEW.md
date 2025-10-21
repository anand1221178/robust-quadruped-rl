# Experiment 5: Per-Joint Deep Dive Analysis

## Overview
Comprehensive individual joint failure analysis for each model with extended metrics, velocity profiling, and anatomical pattern discovery.

## Key Differences from Experiment 3

| Feature | Experiment 3 | Experiment 5 |
|---------|-------------|--------------|
| **Rollouts per joint** | 100 | **150** (+50% more data) |
| **Velocity profiling** | ❌ | ✅ Full trajectory |
| **Fall timing** | ❌ | ✅ Tracks when robot falls |
| **Recovery detection** | ❌ | ✅ Detects recovery attempts |
| **Retention % calculation** | ❌ | ✅ vs baseline performance |
| **Anatomical analysis** | ❌ | ✅ Hip/ankle, camera position |
| **Statistical depth** | Basic | **Comprehensive** (Q25/Q50/Q75) |
| **Best/worst analysis** | ❌ | ✅ Per model |
| **Model ranking** | ❌ | ✅ With medals 🥇🥈🥉 |

## What This Experiment Reveals

### 1. **Retention Percentage Matrix**
Shows exactly what % of baseline performance each model retains for each specific joint:
- **M1 (Baseline)**: How much does it naturally retain?
- **M2 (SR2L)**: Is sensor noise training helpful for joint failures?
- **M3 (DR)**: Which joints benefit most from DR training?
- **M4 (Combo)**: Does combining methods help or hurt?

### 2. **Anatomical Patterns**
Discovers systematic patterns:
- **Hip vs Ankle**: Are hip failures easier to handle than ankle failures?
- **Camera-facing vs Away**: Does viewing angle matter?
- **Front vs Rear**: Are front legs more important for locomotion?

### 3. **Velocity Profiling**
Tracks instantaneous velocity throughout each episode:
- Does the robot maintain speed or slow down?
- Does it try to recover after stumbling?
- When exactly does it fall (if it falls)?

### 4. **Best/Worst Joint Analysis**
For each model, identifies:
- Which joint failure it handles **BEST**
- Which joint failure it handles **WORST**
- The performance gap between them

### 5. **Model Ranking**
Ranks all 4 models by average retention across all 8 joints with medals (🥇🥈🥉)

## Statistics Collected (Per Model-Joint Combination)

### Distance Statistics
- Mean, Std, Min, Max, Median
- Q25 (25th percentile), Q75 (75th percentile)

### Velocity Statistics
- Average velocity per episode
- Max observed velocity
- Min observed velocity

### Performance Metrics
- Success rate (≥1.5m threshold)
- Failure rate (robot falls)
- Retention percentage vs baseline

### Trajectory Statistics
- Early failure rate (before 2-second delay)
- Recovery attempt rate (backward movement after fall)
- Average fall time (when robot falls)
- Median fall time

## Research Questions Answered

### Q1: Is ankle_4 universally hardest?
Or is it only hard for certain models? Experiment 5 shows retention % for every model-joint combination.

### Q2: Does DR training create joint-specific expertise?
By comparing M3's performance across all joints, we can see if it excels at specific types of failures.

### Q3: Do anatomical factors predict failure severity?
Automatic analysis of hip/ankle and camera-facing patterns reveals systematic relationships.

### Q4: How do models adapt under failure?
Velocity profiling shows whether models maintain steady speed, slow down gradually, or stumble and recover.

### Q5: Is there a universal "hard" joint?
Or do different models struggle with different joints? Best/worst analysis reveals model-specific weaknesses.

## Visualizations Generated

### Figure 6: Retention Percentage Matrix
Large heatmap showing retention % for all 32 model-joint combinations with ✓/~/✗ markers:
- ✓ = ≥50% retention (good)
- ~ = 30-50% retention (moderate)
- ✗ = <30% retention (poor)

### Figure 7: Anatomical Pattern Analysis (4 panels)
**Panel 1**: Hip vs Ankle retention comparison across all models
**Panel 2**: Camera-facing vs Camera-away comparison across all models
**Panel 3**: Best vs Worst joint for each model (with joint labels)
**Panel 4**: Overall model ranking with medals 🥇🥈🥉

## Running the Experiment

### Run Experiment 5 Only:
```bash
cd evaluations
python experiment_5_per_joint_deep_dive.py
```

**Estimated Time**: ~2.5 hours
**Total Episodes**: 4,800 (4 models × 8 joints × 150 rollouts)

### Run All 5 Experiments:
```bash
cd evaluations
python run_all_experiments.py
```

**Total Time**: ~20 hours
**Total Episodes**: 15,600

### Run Specific Experiment:
```bash
python run_all_experiments.py 5  # Run only Experiment 5
python run_all_experiments.py --list  # List all experiments
```

## Generate Visualizations

After running experiments:
```bash
cd evaluations
python analyze_and_visualize.py
```

This generates all 7 figures including the 2 new Experiment 5 figures.

## Output Files

### Data:
```
evaluations/experiment_5_per_joint_deep_dive/data/
└── per_joint_deep_dive_results_YYYYMMDD_HHMMSS.json
```

### Figures:
```
evaluations/figures/
├── figure_6_retention_matrix.png
└── figure_7_anatomical_patterns.png
```

## JSON Structure

The results JSON has this structure:
```json
{
  "M1_baseline": [
    {
      "model_key": "M1_baseline",
      "model_name": "PPO Baseline",
      "failed_joint": "hip_1",
      "joint_anatomy": {
        "leg": "front-left",
        "type": "hip",
        "camera_facing": false
      },
      "distance": {
        "mean": 8.35,
        "std": 2.14,
        "min": 3.21,
        "max": 12.67,
        "median": 8.52,
        "q25": 6.78,
        "q75": 9.94
      },
      "velocity": {
        "mean": 0.174,
        "std": 0.045,
        "max_observed": 0.312,
        "min_observed": 0.023
      },
      "success_rate": 0.87,
      "failure_rate": 0.13,
      "baseline_distance": 11.20,
      "retention_percentage": 74.6,
      "trajectory_stats": {
        "early_failure_rate": 0.02,
        "recovery_attempt_rate": 0.18,
        "avg_fall_time_steps": 487.3,
        "median_fall_time_steps": 512.0
      },
      "rollouts": [...]
    },
    ... (7 more joints)
  ],
  ... (3 more models)
}
```

## Key Insights Expected

Based on your previous research:

1. **M3 (DR) should dominate** in average retention across all joints (trained for this)
2. **Ankle_4 will likely be hardest** for all models (confirmed in previous work)
3. **Hip failures should be easier** than ankle failures (more DoF for compensation)
4. **M2 (SR2L) will struggle** more than others (not trained for structural failures)
5. **M4 (Combo) performance unclear** - will synergy appear here or remain negative?

## Research Value

This experiment provides:
- **Publication-ready detailed analysis** of joint-specific robustness
- **Anatomical insights** for future training curriculum design
- **Model comparison** at the finest granularity (32 combinations)
- **Statistical confidence** (150 rollouts vs 100)
- **Velocity profiling** revealing adaptation mechanisms
- **Answers to 5 critical research questions** about joint failure robustness

## Next Steps After Running

1. Review retention matrix (Figure 6) to identify patterns
2. Examine anatomical analysis (Figure 7) for systematic relationships
3. Use best/worst joint data to understand model-specific weaknesses
4. Incorporate velocity profiling insights into Discussion section
5. Use trajectory statistics (early failures, recovery attempts) for mechanism analysis
