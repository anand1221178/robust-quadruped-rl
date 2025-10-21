# Experiment 6: Research Claims Validation Suite

## Purpose
Rigorously validate 4 major discoveries from Experiments 1-4 through controlled ablation studies and statistical analysis.

---

## Why This Experiment Matters

Your Experiments 1-4 revealed **surprising findings**, but they're currently **observations**. Experiment 6 transforms them into **validated scientific claims** with:

1. **Causality**: Ablation studies prove VecNormalize **CAUSES** robustness (not just correlation)
2. **Mechanism**: Stochastic resonance explains **WHY** SR2L improves with noise
3. **Generalizability**: Statistical tests show if patterns hold universally
4. **Publication quality**: p-values, t-tests, ANOVA give reviewers confidence

**This is the difference between**:
- "We noticed X" → "We proved X through controlled experiments" ✅

---

## The 4 Claims Being Validated

### Claim 1: "VecNormalize Provides Implicit Noise Robustness" 😮
**Observation**: M1 (Baseline) maintained 97.8% at 10X training noise despite no robustness training

**Hypothesis**: VecNormalize observation normalization filters sensor noise implicitly

**Test**: Compare M1 **with vs without** VecNormalize under noise

### Claim 2: "Stochastic Resonance in SR2L" 🔥
**Observation**: M2 (SR2L) IMPROVED from 8.91m → 9.00m with noise (101% retention!)

**Hypothesis**: Mild noise helps SR2L through stochastic resonance (neuroscience phenomenon)

**Test**: Fine-grained noise sweep to find optimal noise level where performance peaks

### Claim 3: "Hip_1 Super-Recovery Phenomenon" 🤯
**Observation**: V7.8f walked FASTER with hip_1 locked (105.5% retention!)

**Hypothesis**: Hip_1 failure forces more efficient tripod gait pattern

**Test**: All 4 models with hip_1 locked (300 rollouts for statistical power)

### Claim 4: "Ankle_4 is Universally Hardest" 📍
**Observation**: ALL models struggle most with ankle_4

**Hypothesis**: Ankle_4's anatomical position (rear-camera-facing) creates universal difficulty

**Test**: Statistical ranking of all joints with ANOVA + post-hoc tests

---

## Test Specifications

### Test 1: VecNormalize Ablation Study
**Episodes**: 600 (2 conditions × 3 noise levels × 100 rollouts)
**Time**: ~30 minutes
**Model**: M1 (Baseline)
**Conditions**:
- **With VecNormalize**: Standard evaluation (normalizes observations)
- **Without VecNormalize**: Raw observations (no normalization)

**Noise Levels**: σ = 0.0, 0.05, 0.1

**Expected Result**:
- **IF CLAIM TRUE**: Without VecNormalize, performance crashes at σ=0.1 (<50% retention)
- **IF CLAIM FALSE**: Similar performance with/without VecNormalize

**Analysis**: t-test comparing performance at σ=0.1 (with vs without)

---

### Test 2: Stochastic Resonance Validation
**Episodes**: 700 (7 noise levels × 100 rollouts)
**Time**: ~20 minutes
**Model**: M2 (SR2L)

**Fine-Grained Noise Levels**: σ = 0.000, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100

**Expected Result**:
- **IF CLAIM TRUE**: Performance peak at σ ≈ 0.01-0.02 (mild noise), then gradual decline
- **IF CLAIM FALSE**: Monotonic decrease from σ=0.0 (no peak)

**Analysis**: Find optimal noise level and verify >100% retention

**Stochastic Resonance Explanation**:
> In neuroscience, adding small amounts of noise can actually IMPROVE signal detection by helping weak signals cross activation thresholds. SR2L, trained with noise regularization, may exploit this phenomenon.

---

### Test 3: Hip_1 Super-Recovery Investigation
**Episodes**: 1,200 (4 models × 300 rollouts)
**Time**: ~30 minutes
**Models**: All 4 (M1, M2, M3, M4)

**Test**: Lock hip_1 joint after 2-second delay

**Expected Result**:
- **IF CLAIM TRUE**: At least one model shows >100% retention with hip_1 locked
- **IF CLAIM FALSE**: All models <100% retention (degradation only)

**Analysis**: Rank models by hip_1 robustness, identify super-recovery cases

**Mechanism Hypothesis**:
> Locking hip_1 (front-left hip) may force the robot into a more stable tripod gait where the remaining 3 legs form a more efficient support pattern. This "forced optimization" could accidentally improve locomotion efficiency.

---

### Test 4: Joint Difficulty Ranking (Statistical)
**Episodes**: 0 (uses Experiment 3 data)
**Time**: ~5 minutes (analysis only)
**Data Source**: Experiment 3 results

**Statistical Tests**:
1. **One-way ANOVA**: Test if joint difficulty varies significantly across all 8 joints
2. **Post-hoc t-test**: Test if ankle_4 significantly harder than other joints combined
3. **Universality check**: Is ankle_4 hardest for ALL 4 models individually?

**Expected Result**:
- **IF CLAIM TRUE**:
  - ANOVA p < 0.001 (significant variation)
  - Ankle_4 vs others p < 0.001 (significantly worse)
  - Ankle_4 hardest for 4/4 models (universal)
- **IF CLAIM FALSE**: No significant differences, or other joint hardest

**Joint Difficulty Ranking**:
Joints ranked by average retention percentage across all 4 models (lower = harder)

---

## Output Structure

### JSON Format
```json
{
  "test_1_vecnormalize": {
    "with_vecnormalize": {
      "noise_0.0": {"mean": 11.20, "std": 0.00, "success_rate": 1.0},
      "noise_0.05": {"mean": 11.05, "std": 0.12, "success_rate": 0.99},
      "noise_0.1": {"mean": 10.95, "std": 0.24, "success_rate": 0.98}
    },
    "without_vecnormalize": {
      "noise_0.0": {"mean": 11.18, "std": 0.02, "success_rate": 1.0},
      "noise_0.05": {"mean": 8.45, "std": 1.84, "success_rate": 0.72},
      "noise_0.1": {"mean": 4.23, "std": 2.15, "success_rate": 0.31}
    }
  },

  "test_2_stochastic_resonance": {
    "noise_0.000": {"mean": 8.91, "retention_pct": 100.0},
    "noise_0.005": {"mean": 8.93, "retention_pct": 100.2},
    "noise_0.010": {"mean": 9.02, "retention_pct": 101.2},  // PEAK!
    "noise_0.020": {"mean": 8.98, "retention_pct": 100.8},
    "noise_0.030": {"mean": 8.94, "retention_pct": 100.3},
    "noise_0.050": {"mean": 8.89, "retention_pct": 99.8},
    "noise_0.100": {"mean": 8.54, "retention_pct": 95.9}
  },

  "test_3_hip1_recovery": {
    "M1_baseline": {
      "baseline": 11.20,
      "with_hip1_failure": 8.24,
      "retention_pct": 73.6,
      "super_recovery": false
    },
    "M2_sr2l": {...},
    "M3_dr": {...},
    "M4_combo": {...}
  },

  "test_4_joint_ranking": {
    "ranked_joints": [
      ["ankle_4", 15.2],  // Hardest
      ["ankle_3", 23.4],
      ["hip_4", 31.5],
      ...
      ["hip_1", 74.8]     // Easiest
    ],
    "anova_f": 47.23,
    "anova_p": 0.000001,
    "ankle4_vs_others_t": -12.45,
    "ankle4_vs_others_p": 0.000001,
    "universal_hardest": true
  }
}
```

---

## Expected Findings

### Best Case (All Claims Validated) ✅
1. **VecNormalize**: Retention drops from 98% → 31% without normalization (p < 0.001)
2. **Stochastic Resonance**: Peak at σ=0.01 with 101.2% retention
3. **Hip_1**: At least 1 model shows >100% retention (super-recovery)
4. **Ankle_4**: Hardest for all 4 models (p < 0.001)

**Result**: 4/4 claims validated → Strong Discussion section material

### Partial Validation
1-3 claims validated → Still excellent research contribution

### Null Results ❌
If claims not validated → Document as negative results (also valuable!)
- Shows initial observations were artifacts or model-specific
- Still publishable as "surprising initial observations not generalized"

---

## Research Value

### For Your Paper

**Methods Section**:
- "We performed rigorous ablation studies to validate 4 key observations..."
- Shows scientific rigor beyond descriptive experiments

**Results Section**:
- Use Test 1 to explain unexpected baseline robustness
- Use Test 2 to introduce stochastic resonance phenomenon
- Use Test 3 to show counterintuitive improvement from failure
- Use Test 4 to prove ankle_4 is systematic challenge

**Discussion Section**:
- **VecNormalize finding**: Implications for RL deployment (normalization crucial!)
- **Stochastic resonance**: Connection to neuroscience literature
- **Hip_1 super-recovery**: Forced optimization phenomenon
- **Ankle_4 difficulty**: Anatomical/physics constraints limit all approaches

**Novelty**:
- Stochastic resonance in RL robustness (rare in literature!)
- Quantified VecNormalize contribution (often overlooked)
- Super-recovery from failures (counterintuitive finding)

---

## Running the Experiment

### Run Standalone
```bash
cd evaluations
python experiment_6_validation_suite.py
```

**Estimated Time**: ~2.5 hours
**Total Episodes**: ~2,400

### Run as Part of Suite
```bash
python run_all_experiments.py 6    # Run only Experiment 6
python run_all_experiments.py      # Run all 6 experiments
```

---

## Interpretation Guide

### Test 1: VecNormalize Impact

**Strong Support** (retention difference >30%):
> "VecNormalize provides substantial implicit noise robustness, reducing performance degradation from 69% to 2% at σ=0.1 (p < 0.001). This explains why all models maintained high performance despite sensor noise."

**Moderate Support** (10-30% difference):
> "VecNormalize contributes moderately to noise robustness..."

**Weak/No Support** (<10% difference):
> "VecNormalize provides minimal noise filtering; robustness likely comes from other sources."

### Test 2: Stochastic Resonance

**Confirmed** (peak >100% retention):
> "SR2L exhibits stochastic resonance, with optimal performance at σ=0.01 (101.2% retention). This neuroscience phenomenon suggests mild noise aids SR2L's observation processing."

**Not Confirmed** (no peak >100%):
> "While SR2L maintains performance under noise, no stochastic resonance effect observed."

### Test 3: Super-Recovery

**Confirmed** (any model >100%):
> "Model X demonstrates super-recovery from hip_1 failure (105% retention), suggesting the constraint forces a more efficient gait pattern."

**Not Confirmed** (all <100%):
> "No super-recovery observed in current models. Original finding (V7.8f) may have been model-specific."

### Test 4: Ankle_4 Difficulty

**Universally Hardest** (p < 0.001, 4/4 models):
> "Ankle_4 is statistically the hardest joint across all models (ANOVA p < 0.001), confirming this is an anatomical/physics constraint rather than training artifact."

**Hardest on Average** (p < 0.001, but not universal):
> "Ankle_4 is significantly harder on average, though some models show different worst joints."

---

## Statistical Thresholds

**Highly Significant**: p < 0.001 ✅
**Significant**: p < 0.05 ✓
**Trending**: p < 0.10 ~
**Not Significant**: p ≥ 0.10 ✗

**Effect Size** (Cohen's d):
- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8
- Very Large: d > 1.2

---

## Citation Template

If validated findings are used:

```bibtex
@article{patel2025validation,
  title={Validation of Implicit Robustness Mechanisms in Robust RL},
  author={Patel, Anand},
  journal={Under Review},
  year={2025},
  note={VecNormalize ablation, stochastic resonance, super-recovery phenomena}
}
```

---

**Last Updated**: October 13, 2025
**Status**: Ready to run
**Expected Completion**: ~2.5 hours
**Research Impact**: High (validates surprising discoveries)
