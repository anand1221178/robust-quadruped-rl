#!/usr/bin/env python3
"""
Compute Statistical Significance for Paper Results
Generates p-values, confidence intervals, and effect sizes
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Bonferroni correction for multiple comparisons
ALPHA = 0.05

def load_json(filepath: Path) -> Dict:
    """Load JSON data file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def paired_t_test_with_bonferroni(data1: List[float], data2: List[float],
                                   n_comparisons: int) -> Tuple[float, bool, str]:
    """
    Perform paired t-test with Bonferroni correction
    Returns: (p-value, is_significant, significance_marker)
    """
    t_stat, p_value = stats.ttest_rel(data1, data2)
    bonferroni_alpha = ALPHA / n_comparisons
    is_significant = p_value < bonferroni_alpha

    if p_value < 0.001:
        marker = "***"
    elif p_value < 0.01:
        marker = "**"
    elif p_value < bonferroni_alpha:
        marker = "*"
    else:
        marker = "ns"

    return p_value, is_significant, marker

def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval"""
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return ci

def cohen_d(data1: List[float], data2: List[float]) -> float:
    """Compute Cohen's d effect size"""
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / pooled_std

# ============================================================================
# EXPERIMENT 1: BASELINE PERFORMANCE
# ============================================================================
print("="*80)
print("EXPERIMENT 1: BASELINE PERFORMANCE")
print("="*80)

exp1_file = Path("../evaluations/evaluations/experiment_1_baseline/data/baseline_results_20251027_195743.json")
exp1_data = load_json(exp1_file)

models = ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M1 (Baseline)', 'M2 (SR2L)', 'M3 (DR)', 'M4 (Combined)']

print("\n--- Baseline Distance Statistics ---")
baseline_distances = {}
for model, label in zip(models, model_labels):
    distances = [ep['distance'] for ep in exp1_data[model]['rollouts']]
    baseline_distances[model] = distances

    mean = np.mean(distances)
    std = np.std(distances, ddof=1)
    ci = compute_confidence_interval(distances)

    print(f"{label:20s}: {mean:6.2f} ± {std:5.2f}m  (95% CI: [{ci[0]:6.2f}, {ci[1]:6.2f}])")

# Pairwise comparisons (6 comparisons total for 4 models)
print("\n--- Pairwise Statistical Comparisons (Bonferroni corrected) ---")
n_comparisons = 6  # C(4,2) = 6 pairwise comparisons

comparisons = [
    ('M1_baseline', 'M2_sr2l'),
    ('M1_baseline', 'M3_dr'),
    ('M1_baseline', 'M4_combo'),
    ('M2_sr2l', 'M3_dr'),
    ('M2_sr2l', 'M4_combo'),
    ('M3_dr', 'M4_combo'),
]

for model1, model2 in comparisons:
    label1 = model_labels[models.index(model1)]
    label2 = model_labels[models.index(model2)]

    p_val, sig, marker = paired_t_test_with_bonferroni(
        baseline_distances[model1],
        baseline_distances[model2],
        n_comparisons
    )

    effect_size = cohen_d(baseline_distances[model1], baseline_distances[model2])

    print(f"{label1} vs {label2}: p={p_val:.4f} {marker}  (Cohen's d={effect_size:+.2f})")

# ============================================================================
# EXPERIMENT 2: SENSOR NOISE ROBUSTNESS
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: SENSOR NOISE ROBUSTNESS")
print("="*80)

exp2_file = Path("../evaluations/evaluations/experiment_2_sensor_noise/data/sensor_noise_results_20251027_201704.json")
exp2_data = load_json(exp2_file)

# Test at key noise level: σ=0.10 (10x training noise for M2)
print("\n--- Performance at σ=0.10 (10× M2 training noise) ---")
noise_level = 0.10

for model, label in zip(models, model_labels):
    # Get baseline (noise_level=0.0)
    baseline_result = [r for r in exp2_data[model]['noise_results'] if r['noise_level'] == 0.0][0]
    baseline_dist = baseline_result['distance']['mean']

    # Get noise result
    noise_results = [r for r in exp2_data[model]['noise_results'] if abs(r['noise_level'] - noise_level) < 0.001]
    if noise_results:
        noise_dist = noise_results[0]['distance']['mean']
        noise_std = noise_results[0]['distance']['std']
        retention = (noise_dist / baseline_dist) * 100

        print(f"{label:20s}: {noise_dist:6.2f} ± {noise_std:5.2f}m  ({retention:5.1f}% retention)")

# Test significance: Does noise degrade performance?
print("\n--- Noise Impact Statistical Tests (σ=0.10 vs baseline) ---")
for model, label in zip(models, model_labels):
    baseline_eps = baseline_distances[model]

    noise_results = [r for r in exp2_data[model]['noise_results'] if abs(r['noise_level'] - noise_level) < 0.001]
    if noise_results:
        noise_eps = [ep['distance'] for ep in noise_results[0]['rollouts']]

        # Only test if we have matching episode counts
        if len(baseline_eps) == len(noise_eps):
            p_val, sig, marker = paired_t_test_with_bonferroni(baseline_eps, noise_eps, 4)
            effect_size = cohen_d(baseline_eps, noise_eps)

            degradation = ((np.mean(baseline_eps) - np.mean(noise_eps)) / np.mean(baseline_eps)) * 100

            print(f"{label}: {degradation:+5.1f}% degradation, p={p_val:.4f} {marker} (d={effect_size:+.2f})")
        else:
            print(f"{label}: Cannot compute (different sample sizes)")

# ============================================================================
# EXPERIMENT 3: JOINT FAILURE ROBUSTNESS
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 3: JOINT FAILURE ROBUSTNESS")
print("="*80)

exp3_file = Path("../evaluations/evaluations/experiment_3_joint_failures/data/joint_failure_results_20251027_205918.json")
exp3_data = load_json(exp3_file)

print("\n--- Average Retention Across All 8 Joints ---")
joint_retentions = {}

for model, label in zip(models, model_labels):
    retentions = []
    for joint_result in exp3_data[model]['joint_results']:
        retention = joint_result['retention_percentage']
        retentions.append(retention)

    joint_retentions[model] = retentions
    mean_ret = np.mean(retentions)
    std_ret = np.std(retentions, ddof=1)

    print(f"{label:20s}: {mean_ret:5.1f}% ± {std_ret:4.1f}%")

# Statistical comparison: M3 vs M4 (key finding)
print("\n--- Key Comparison: M3 (DR) vs M4 (Combined) ---")
p_val, sig, marker = paired_t_test_with_bonferroni(
    joint_retentions['M3_dr'],
    joint_retentions['M4_combo'],
    1  # Single focused comparison
)
effect_size = cohen_d(joint_retentions['M3_dr'], joint_retentions['M4_combo'])

print(f"M3 vs M4: p={p_val:.4f} {marker}  (Cohen's d={effect_size:+.2f})")
print(f"Conclusion: M3 {'significantly' if sig else 'not significantly'} better than M4")

# Hip vs Ankle analysis
print("\n--- Hip vs Ankle Comparison ---")
joints_hip = ['hip_1', 'hip_2', 'hip_3', 'hip_4']
joints_ankle = ['ankle_1', 'ankle_2', 'ankle_3', 'ankle_4']

for model, label in zip(models, model_labels):
    hip_rets = []
    ankle_rets = []

    for joint_result in exp3_data[model]['joint_results']:
        joint_name = joint_result['joint']
        retention = joint_result['retention_percentage']

        if joint_name in joints_hip:
            hip_rets.append(retention)
        elif joint_name in joints_ankle:
            ankle_rets.append(retention)

    p_val, sig, marker = paired_t_test_with_bonferroni(hip_rets, ankle_rets, 4)

    print(f"{label}: Hip={np.mean(hip_rets):.1f}%, Ankle={np.mean(ankle_rets):.1f}%, p={p_val:.4f} {marker}")

# ============================================================================
# EXPERIMENT 4: COMBINED STRESS (KEY FINDING)
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 4: COMBINED STRESS (M4 UNDERPERFORMANCE)")
print("="*80)

exp4_file = Path("../evaluations/evaluations/experiment_4_combined_stress/data/combined_stress_results_20251013_112858.json")
exp4_data = load_json(exp4_file)

# Find the highest stress scenario
print("\n--- Highest Stress Scenario: 50% joint failure + σ=0.05 noise ---")
high_stress_distances = {}

for model, label in zip(models, model_labels):
    # Find scenario with 50% failure rate and σ=0.05
    for scenario in exp4_data[model]['scenario_results']:
        if scenario['joint_failure_prob'] == 0.5 and scenario['sensor_noise'] == 0.05:
            distances = [ep['distance'] for ep in scenario['rollouts']]
            high_stress_distances[model] = distances

            mean = np.mean(distances)
            std = np.std(distances, ddof=1)

            # Get baseline from experiment 1
            baseline = np.mean(baseline_distances[model])
            retention = (mean / baseline) * 100

            print(f"{label:20s}: {mean:5.2f} ± {std:4.2f}m  ({retention:5.1f}% retention)")

# THE KEY STATISTICAL TEST: M3 vs M4 under combined stress
print("\n--- CRITICAL COMPARISON: M3 vs M4 under Combined Stress ---")
if 'M3_dr' in high_stress_distances and 'M4_combo' in high_stress_distances:
    p_val, sig, marker = paired_t_test_with_bonferroni(
        high_stress_distances['M3_dr'],
        high_stress_distances['M4_combo'],
        1
    )
    effect_size = cohen_d(high_stress_distances['M3_dr'], high_stress_distances['M4_combo'])

    m3_mean = np.mean(high_stress_distances['M3_dr'])
    m4_mean = np.mean(high_stress_distances['M4_combo'])
    underperformance = ((m3_mean - m4_mean) / m3_mean) * 100

    print(f"M3 (DR): {m3_mean:.2f}m")
    print(f"M4 (Combo): {m4_mean:.2f}m")
    print(f"M4 underperforms M3 by: {underperformance:.1f}%")
    print(f"Statistical significance: p={p_val:.6f} {marker}")
    print(f"Effect size: Cohen's d={effect_size:+.2f}")

    if sig:
        print(" FINDING CONFIRMED: M4 significantly worse than M3 (p<0.05)")
    else:
        print("⚠️  WARNING: Difference not statistically significant")

print("\n" + "="*80)
print("STATISTICAL ANALYSIS COMPLETE")
print("="*80)
