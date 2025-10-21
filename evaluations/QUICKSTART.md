# Quick Start Guide - Evaluation Suite

## Prerequisites

✅ All 4 models trained and saved in `done/` directory:
- `done/ppo_baseline_ueqbjf2x/` (M1: PPO Baseline)
- `done/ppo_sr2l_forward_m7gtjtpa/` (M2: PPO + SR2L)
- `done/v7_7e_ultra_speed_jtfwl2qf/` (M3: PPO + DR)
- `done/ultimate_robustness_combo_ju7lfsk2/` (M4: Ultimate Combo)

✅ Python environment with required packages installed

---

## Option 1: Run All Experiments (Recommended)

**Full evaluation suite - all 4 experiments in sequence**

```bash
cd evaluations
python run_all_experiments.py
```

**What it does:**
- Runs all 4 experiments sequentially
- Total runtime: ~14-15 hours
- Total episodes: 9,200
- Saves results to separate JSON files

**When to use:**
- Running overnight or on cluster
- Want complete evaluation for paper
- Have 14-15 hours available

---

## Option 2: Run Individual Experiments

**Run specific experiment by number**

```bash
cd evaluations

# Experiment 1: Baseline (40 min, 400 episodes)
python run_all_experiments.py 1

# Experiment 2: Sensor Noise (5 hrs, 3200 episodes)
python run_all_experiments.py 2

# Experiment 3: Joint Failures (5 hrs, 3200 episodes)
python run_all_experiments.py 3

# Experiment 4: Combined Stress (4 hrs, 2400 episodes)
python run_all_experiments.py 4
```

**When to use:**
- Testing one aspect at a time
- Limited time available
- Need quick results for specific test

---

## Option 3: Run Experiment Scripts Directly

**Direct execution for maximum control**

```bash
cd evaluations

python experiment_1_baseline.py
python experiment_2_sensor_noise.py
python experiment_3_joint_failures.py
python experiment_4_combined_stress.py
```

**When to use:**
- Debugging individual experiments
- Modifying experiment parameters
- Custom analysis workflow

---

## Recommended Workflow

### For Supervisor Meeting
```bash
# Quick baseline check (40 min)
cd evaluations
python run_all_experiments.py 1
```
Review baseline results before committing to full evaluation.

### For Full Evaluation
```bash
# Run overnight (14-15 hours)
cd evaluations
nohup python run_all_experiments.py > evaluation_log.txt 2>&1 &
```
Check progress: `tail -f evaluation_log.txt`

### For Paper Figures
```bash
# Run all experiments
cd evaluations
python run_all_experiments.py

# Then analyze results
cd ../
python scripts/analyze_results.py  # Create this next
```

---

## Quick Test (5 minutes)

Want to verify everything works before full run?

**Modify `num_rollouts` in any experiment script:**

```python
# Change from:
self.num_rollouts = 100

# To:
self.num_rollouts = 5  # Quick test
```

Then run:
```bash
python experiment_1_baseline.py
```

Should complete in ~2 minutes and verify:
- Models load correctly
- Environments work
- Metrics calculate properly
- Results save to JSON

---

## What to Expect

### Experiment 1 Output
```
================================================================================
EXPERIMENT 1: BASELINE PERFORMANCE EVALUATION
================================================================================
Testing all 4 models under ideal conditions
- No sensor noise
- No joint failures
- Episode length: 1200 steps (20 seconds)
- Rollouts per model: 100
...

============================================================
Evaluating: PPO Baseline
============================================================
Running 100 rollouts...
100%|████████████████████████| 100/100 [00:10<00:00, 9.5it/s]

============================================================
RESULTS: PPO Baseline
============================================================
Distance:     10.545 ± 2.341 m
Reward:       347.2 ± 89.5
Success Rate: 85.0% (85/100 episodes)
Failure Rate: 5.0% (5/100 episodes)
...

✅ Results saved to: evaluations/experiment_1_baseline/data/baseline_results_20251013_093045.json
```

### Final Summary
```
================================================================================
BASELINE PERFORMANCE COMPARISON
================================================================================
Model                     Distance (m)    Success Rate    Failure Rate
--------------------------------------------------------------------------------
PPO Baseline              10.545 ± 2.341   85.0%           5.0%
PPO + SR2L                 9.821 ± 2.156   82.0%           6.0%
PPO + DR (V7.7E)           8.932 ± 2.789   78.0%           8.0%
Ultimate Combo             9.445 ± 2.234   80.0%           7.0%
================================================================================

KEY FINDINGS:
✅ Best Distance:    PPO Baseline (10.545m)
✅ Best Success:     PPO Baseline (85.0%)
✅ Most Stable:      PPO Baseline (5.0% failure)
```

---

## Results Location

All results saved to timestamped JSON files:

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
└── experiment_4_combined_stress/
    └── data/
        └── combined_stress_results_20251013_193721.json
```

---

## Troubleshooting

### "Model not found" error
```bash
# Check model paths
ls -la done/ppo_baseline_ueqbjf2x/best_model/
ls -la done/ppo_sr2l_forward_m7gtjtpa/

# Verify paths in experiment scripts match your structure
```

### "Out of memory" error
- Reduce `num_rollouts` from 100 to 50
- Close other applications
- Run one experiment at a time

### Slow execution
- Expected: 1.5-2 episodes/second
- If much slower: check CPU usage, close background apps
- Consider running on cluster with more resources

### Want to stop early
- Press `Ctrl+C` to interrupt
- Already completed results will be saved
- Can resume by running remaining experiments individually

---

## Next Steps After Completion

1. **Check all result files exist:**
   ```bash
   ls -lh evaluations/*/data/*.json
   ```

2. **Quick result check:**
   ```bash
   python -c "import json; print(json.load(open('evaluations/experiment_1_baseline/data/baseline_results_20251013_093045.json'))['M1_baseline']['distance'])"
   ```

3. **Statistical analysis:**
   - Load all JSONs into analysis script
   - Run t-tests, ANOVA
   - Test hypotheses H1, H2, H3

4. **Generate figures:**
   - Distance comparison bar charts
   - Noise robustness curves
   - Joint failure heatmaps
   - Combined stress results

---

## Time Budget

| Experiment | Runtime | Episodes | When to Run |
|------------|---------|----------|-------------|
| Exp 1: Baseline | 40 min | 400 | Quick baseline check |
| Exp 2: Noise | 5 hrs | 3,200 | Overnight or afternoon |
| Exp 3: Failures | 5 hrs | 3,200 | Overnight or afternoon |
| Exp 4: Combined | 4 hrs | 2,400 | Evening or afternoon |
| **Total** | **~15 hrs** | **9,200** | **Full overnight run** |

---

## Getting Help

- **README.md** - Detailed documentation
- **EVALUATION_PLAN.md** - Research methodology
- **CLAUDE.md** - Project context and history

Questions? Check these files first!

---

*Ready to start? Run:* `cd evaluations && python run_all_experiments.py`
