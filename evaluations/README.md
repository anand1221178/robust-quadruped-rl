# Evaluation Suite for Robust Quadruped Locomotion Research

**Complete 4-Way Ablation Study Implementation**

This directory contains the comprehensive evaluation framework for testing the hypothesis that **SR2L and Domain Randomization contribute independently to robustness**.

---

## Overview

### Research Question
> Can a quadruped locomotion policy trained using SR2L and/or Domain Randomization achieve robustness to sensor noise and actuator failures, and do these methods contribute independently?

### 4-Way Ablation Study

| Model ID | Name | Components | Specialty |
|----------|------|------------|-----------|
| **M1** | PPO Baseline | PPO only | None (baseline performance) |
| **M2** | PPO + SR2L | PPO + Smooth Regularization | **Sensor noise specialist** |
| **M3** | PPO + DR | PPO + Domain Randomization | **Joint failure specialist** |
| **M4** | Ultimate Combo | PPO + SR2L + DR | **Combined robustness** |

---

## Experiments

### Experiment 1: Baseline Performance
**File:** `experiment_1_baseline.py`

**Purpose:** Establish baseline performance under ideal conditions

**Test Conditions:**
- No sensor noise
- No joint failures
- 20-second episodes (1200 steps @ 60fps)
- 100 rollouts per model

**Metrics:**
- Total distance traveled (meters)
- Success rate (% episodes achieving ≥1.5m)
- Failure rate (% episodes with robot collapse)

**Expected Results:**
- All models should perform well
- M1 (baseline) may have highest clean performance
- Tests if robustness training sacrifices baseline performance

**Runtime:** ~40 minutes (400 total episodes)

---

### Experiment 2: Sensor Noise Robustness
**File:** `experiment_2_sensor_noise.py`

**Purpose:** Test sensor noise robustness (SR2L's specialty)

**Test Conditions:**
- 8 noise levels: [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
- Noise applied to joint sensors only (obs dims 13-28)
- Matches SR2L training methodology
- 100 rollouts per noise level per model

**Metrics:**
- Distance at each noise level
- Success rate degradation curve
- Noise tolerance threshold

**Expected Results:**
- **M2 (SR2L) should dominate** at high noise levels
- M4 should match or exceed M2 performance
- M1 and M3 should degrade significantly with noise

**Key Analysis:**
- Does M2 outperform M1 under noise? (tests H1)
- Quantify SR2L's independent contribution

**Runtime:** ~5 hours (3,200 total episodes)

---

### Experiment 3: Joint Failure Robustness
**File:** `experiment_3_joint_failures.py`

**Purpose:** Test actuator failure robustness (DR's specialty)

**Test Conditions:**
- All 8 individual joints tested: hip_1, ankle_1, ..., hip_4, ankle_4
- Delayed locking: 2-second delay before joint locks at 0.0
- Matches DR training methodology
- 100 rollouts per joint per model

**Metrics:**
- Distance with each joint failure
- Success rate per joint
- Average robustness across all joints

**Expected Results:**
- **M3 (DR) should dominate** with joint failures
- M4 should match or exceed M3 performance
- M1 and M2 should struggle significantly

**Key Analysis:**
- Does M3 outperform M1 with failures? (tests H2)
- Quantify DR's independent contribution

**Runtime:** ~5 hours (3,200 total episodes)

---

### Experiment 4: Combined Stress
**File:** `experiment_4_combined_stress.py`

**Purpose:** Test synergy between SR2L and DR (M4's specialty)

**Test Scenarios:**
1. **Mild Combined:** 0.05 noise + Ankle_2 failure
2. **Moderate Combined:** 0.10 noise + Hip_4 failure
3. **Challenging Combined:** 0.10 noise + Ankle_3 failure
4. **Severe Combined:** 0.20 noise + Hip_1 failure
5. **Extreme Dual:** 0.05 noise + Hip_1 + Ankle_2 failures
6. **Ultimate Challenge:** 0.10 noise + Hip_4 + Ankle_3 failures

**Metrics:**
- Distance under combined stress
- Success rate per scenario
- Synergy detection: M4 > max(M2, M3)

**Expected Results:**
- **M4 should outperform both M2 and M3**
- Demonstrates synergistic effect
- Neither M2 nor M3 alone sufficient

**Key Analysis:**
- Does M4 exceed best specialist? (tests H3)
- Quantify synergy magnitude

**Runtime:** ~4 hours (2,400 total episodes)

---

## Running Experiments

### Run All Experiments (Recommended)
```bash
cd evaluations
python run_all_experiments.py
```

**Total Runtime:** ~14-15 hours
**Total Episodes:** 9,200

### Run Individual Experiment
```bash
cd evaluations
python run_all_experiments.py 1  # Experiment 1
python run_all_experiments.py 2  # Experiment 2
python run_all_experiments.py 3  # Experiment 3
python run_all_experiments.py 4  # Experiment 4
```

### Run Experiment Directly
```bash
cd evaluations
python experiment_1_baseline.py
python experiment_2_sensor_noise.py
python experiment_3_joint_failures.py
python experiment_4_combined_stress.py
```

---

## Results Organization

```
evaluations/
├── experiment_1_baseline/
│   └── data/
│       └── baseline_results_YYYYMMDD_HHMMSS.json
├── experiment_2_sensor_noise/
│   └── data/
│       └── sensor_noise_results_YYYYMMDD_HHMMSS.json
├── experiment_3_joint_failures/
│   └── data/
│       └── joint_failure_results_YYYYMMDD_HHMMSS.json
└── experiment_4_combined_stress/
    └── data/
        └── combined_stress_results_YYYYMMDD_HHMMSS.json
```

### Result Format

Each JSON file contains:
```json
{
  "M1_baseline": {
    "model_name": "PPO Baseline",
    "model_key": "M1_baseline",
    "results": [...],  // Per-condition results
    "distance": {
      "mean": 10.5,
      "std": 2.3,
      "min": 5.2,
      "max": 15.8
    },
    "success_rate": 0.85,
    "failure_rate": 0.05
  },
  // ... M2, M3, M4 ...
}
```

---

## Hypothesis Testing

### H1: SR2L Contributes Independently to Sensor Noise Robustness
**Test:** M2 > M1 under high noise (p < 0.05)
**Data:** Experiment 2 results at noise ≥ 0.1
**Analysis:** Paired t-test comparing M2 vs M1 distances

### H2: DR Contributes Independently to Joint Failure Robustness
**Test:** M3 > M1 under joint failures (p < 0.05)
**Data:** Experiment 3 results across all joints
**Analysis:** Paired t-test comparing M3 vs M1 distances

### H3: Combined Approach Achieves Best Overall Robustness
**Test:** M4 ≥ max(M2, M3) under combined stress (p < 0.05)
**Data:** Experiment 4 results across all scenarios
**Analysis:** Compare M4 vs max(M2, M3) to detect synergy

---

## Key Metrics Explained

### Distance Traveled (meters)
- Primary performance metric
- Net forward displacement from start to end of episode
- Higher is better

### Success Rate (%)
- Percentage of episodes achieving ≥1.5m travel distance
- Binary success criterion
- Indicates reliable locomotion

### Failure Rate (%)
- Percentage of episodes where robot falls (torso height < 0.2m)
- Lower is better
- Indicates stability and fall prevention

---

## Expected Outcomes

### Experiment 1: Baseline
- M1 likely highest (no robustness overhead)
- M2, M3, M4 should be within 10-20% of M1
- Validates no catastrophic trade-offs

### Experiment 2: Sensor Noise
- **M2 should dominate at high noise**
- M1, M3 should degrade significantly
- M4 should match M2 (inherits SR2L robustness)

### Experiment 3: Joint Failures
- **M3 should dominate with failures**
- M1, M2 should struggle significantly
- M4 should match M3 (inherits DR robustness)

### Experiment 4: Combined Stress
- **M4 should outperform M2 and M3**
- Demonstrates synergistic effect
- Validates full pipeline superiority

---

## Statistical Analysis (Post-Experiments)

After running all experiments:

1. **Load all result JSONs**
2. **Descriptive statistics:** Mean, std, confidence intervals
3. **Comparative tests:**
   - Paired t-tests between model pairs
   - ANOVA across all 4 models
   - Bonferroni correction for multiple comparisons
4. **Hypothesis tests:** H1, H2, H3 as defined above
5. **Generate figures:**
   - Distance comparison bar charts
   - Success rate curves
   - Robustness degradation plots

---

## Technical Details

### Environment Setup
```python
base_env = gym.make('RealAntMujoco-v0')
env = SuccessRewardWrapper(base_env)
env = TimeLimit(env, max_episode_steps=1200)
env = DummyVecEnv([lambda: env])
env = VecNormalize.load(vec_normalize_path, env)
```

### Sensor Noise Application
- Applied to **joint sensors only** (obs dims 13-28)
- Gaussian noise: `N(0, σ)` where σ ∈ [0.0, 0.7]
- Matches SR2L training distribution

### Joint Failure Application
- **Delayed locking:** 120-step (2-second) delay before lock
- Lock value: 0.0 (neutral position)
- Matches DR training methodology

### Episode Parameters
- Length: 1200 steps (20 seconds @ 60fps)
- Deterministic policy evaluation
- Fixed random seed (42) for reproducibility

---

## Troubleshooting

### Out of Memory
- Reduce `num_rollouts` from 100 to 50
- Run experiments one at a time instead of all at once

### Slow Execution
- Expected: ~14-15 hours total
- Run overnight or on cluster
- Monitor with `tqdm` progress bars

### Model Loading Errors
- Check model paths in script headers
- Ensure all 4 models exist in `../done/` directory
- Verify VecNormalize files present

---

## Questions for Supervisor

Based on EVALUATION_PLAN.md:

1. Is 100 rollouts per condition sufficient for statistical power?
2. Should we test additional noise levels between 0.1-0.3?
3. Are the combined stress scenarios comprehensive enough?
4. What correction method for multiple comparisons (Bonferroni vs Holm)?
5. Should we prioritize any stretch goals (gait analysis, generalization)?

---

## Summary

This evaluation suite provides:

✅ **Complete 4-way ablation study**
✅ **Independent contribution testing** (SR2L and DR)
✅ **Synergy detection** (combined approach)
✅ **Statistical rigor** for publication
✅ **9,200 episodes** across 4 comprehensive experiments
✅ **~15 hours runtime** (feasible for single overnight run)

**Next Steps:**
1. Run evaluation suite
2. Analyze results
3. Generate figures
4. Write Results & Discussion chapters

---

*Last Updated: October 13, 2025*
