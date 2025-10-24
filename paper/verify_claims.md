# Claim Verification for Results Section

## Summary from Statistical Analysis

Based on the statistical tests run on our data:

### ✅ VERIFIED CLAIMS

#### 1. "Conflicting gradients" / "Interference effects"
**RECOMMENDATION**: Use "interference effects" in Results, save "conflicting gradients" for Discussion.

**Evidence**:
- M4 (Combined) has SIGNIFICANTLY lower baseline than M3 (DR): 5.34m vs 7.96m (p<0.001, Cohen's d=+0.84)
- Under sensor noise (σ=0.10), M4 performs worse than all other models
- This is empirical observation of interference, not direct gradient measurement

**Results language**: "M4 exhibited significantly lower performance..."
**Discussion language**: "This interference likely stems from conflicting gradients..."

#### 2. Statistical Significance - NEED TO COMPUTE PROPERLY
**STATUS**: Partially computed

**What we have**:
- Baseline comparisons: All pairwise differences are statistically significant (p<0.001)
- M1 significantly better than M2, M3, M4 (all p<0.001)
- M3 significantly better than M4 (p<0.001, Cohen's d=+0.84)
- Bonferroni correction applied (α=0.05/6 = 0.0083 for 6 comparisons)

**What we need**:
- Joint failure retention calculations (retention_percentage not in raw data - need to compute)
- Combined stress M3 vs M4 comparison
- Hip vs ankle statistical tests

**ACTION**: Need to finish the statistics script with proper retention calculations

#### 3. "VecNormalize acts as low-pass filter"
**STATUS**: NEED EXPERIMENT 6 DATA

**Claim in paper**: "VecNormalize's implicit low-pass filtering"

**Evidence needed**:
- experiment_6_validation_suite should have VecNormalize ablation
- Need to check if we actually ran this test
- Alternative: Soften language to "VecNormalize provides implicit robustness" (descriptive, not mechanistic)

**RECOMMENDATION**: Check experiment_6 data, or cite VecNormalize paper + describe empirical observation

####  4. "Stochastic resonance"
**STATUS**: STRONG MECHANISTIC CLAIM - NEEDS CITATIONS

**Claim in paper**: "SR2L policies exhibit noise-induced regularization, achieving 101% performance"

**Evidence**:
- We observe M2 (SR2L) maintaining/exceeding baseline at certain noise levels
- M2 at σ=0.10: 90.9% retention (slight degradation, not improvement)
- Need to check other noise levels for the 101% claim

**RECOMMENDATION**:
- Results: Report the empirical observation ("M2 maintained performance across noise levels")
- Discussion: Cite stochastic resonance literature from neuroscience/signal processing
- Do NOT claim we discovered stochastic resonance - claim we observed phenomenon consistent with it

**Citations needed**:
- Gammaitoni et al. (1998) "Stochastic resonance" Reviews of Modern Physics
- McDonnell & Abbott (2009) "What is stochastic resonance?" PLOS Computational Biology

#### 5. Camera-facing vs camera-away
**STATUS**: NEED TO COMPUTE FROM DATA

**Claim**: Anatomical patterns based on camera position

**What we need**:
- Joint positions in robot coordinate frame
- Group joints by: camera-facing (joints 2,4?) vs camera-away (joints 1,3?)
- Statistical comparison of retention percentages

**ACTION**: Need to define which joints are camera-facing and compute statistics

**RECOMMENDATION**: Only include if we can clearly define and statistically verify the pattern

---

## ACTION ITEMS BEFORE WRITING RESULTS

1. ✅ **Baseline statistics**: DONE - all models statistically different
2. ⏳ **Joint failure retention**: Need to compute from raw distances
3. ⏳ **Combined stress M3 vs M4**: Need to find correct scenario in experiment 4 data
4. ⏳ **Check experiment_6**: VecNormalize ablation data
5. ⏳ **Verify 101% retention claim**: Check all noise levels for M2
6. ⏳ **Camera position analysis**: Define joint grouping and compute

---

## RECOMMENDED CLAIMS FOR RESULTS (Conservative)

### Baseline Performance (Figure 1)
✅ "M1 achieved significantly higher baseline performance (11.20 ± 0.16m) compared to M2 (8.75 ± 2.35m, p<0.001), M3 (7.96 ± 2.33m, p<0.001), and M4 (5.34 ± 3.75m, p<0.001)."

### Sensor Noise (Figure 2)
✅ "All models maintained >90% performance retention at σ=0.10 (10× M2 training noise): M1: 97.8%, M2: 90.9%, M3: 96.8%."

⚠️ "M4 showed anomalous performance characteristics under sensor noise." (Need to investigate the 213% retention - data issue?)

### Joint Failures (Figure 4)
⏳ Need retention calculations

### Combined Stress (Figure 3)
⏳ Need to verify M3 vs M4 comparison with proper statistical test

### Joint-Noise Interaction (Figure 5)
⏳ Need full factorial analysis

---

## DECISION NEEDED

**Should I**:
A) Fix the statistics script to compute all missing values (30-60 min work)
B) Write Results with the verified claims we have + placeholder language for unverified claims
C) You tell me which specific claims are most important and I prioritize those

**Your call!**
