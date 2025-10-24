# Paper Notes - Statistical Verification & Claims

## 5 KEY CLAIMS TO VERIFY

### 1. "Conflicting gradients" vs "Interference effects"
**DECISION**:
- ✅ Use "interference effects" in Results (factual observation)
- Save "conflicting gradients" for Discussion (mechanistic reasoning)

**Evidence**:
- M4 (Combined) significantly worse than M3 (DR): 5.34m vs 7.96m (p<0.001, Cohen's d=+0.84)
- Empirical observation of interference, not direct gradient measurement

**Results language**: "M4 exhibited significantly lower performance compared to M3..."
**Discussion language**: "This interference likely stems from conflicting training gradients..."

---

### 2. Statistical Significance - COMPUTE ALL TESTS
**STATUS**: In progress - computing all p-values with Bonferroni correction

**Completed**:
- ✅ Baseline comparisons: All pairwise differences significant (p<0.001)
- ✅ M1 > M2, M3, M4 (all p<0.001)
- ✅ M3 > M4 baseline (p<0.001, Cohen's d=+0.84)
- ✅ Bonferroni correction applied (α=0.05/n_comparisons)

**To compute**:
- ⏳ Joint failure retention percentages (need to calculate from raw distances)
- ⏳ Combined stress M3 vs M4 statistical test
- ⏳ Hip vs ankle comparisons
- ⏳ Front vs rear leg comparisons
- ⏳ Camera-facing vs camera-away analysis

---

### 3. "VecNormalize acts as low-pass filter"
**DECISION**: Soften to empirical observation + cite VecNormalize paper

**Current claim**: "VecNormalize's implicit low-pass filtering"

**Revised approach**:
- Results: "VecNormalize normalization provided implicit robustness benefits across all models"
- Discussion: "This normalization acts as an implicit low-pass filter [cite VecNormalize paper], attenuating high-frequency noise"

**Citations needed**:
- OpenAI Baselines VecNormalize documentation/paper
- OR empirical observation language without mechanistic claim

**ACTION**: Check experiment_6_validation_suite for VecNormalize ablation data

---

### 4. "Stochastic resonance" - NEEDS CITATIONS
**DECISION**: Describe empirical observation + cite neuroscience literature

**Current claim**: "SR2L policies exhibit noise-induced regularization, achieving 101% performance"

**Evidence**:
- M2 (SR2L) maintains performance across noise levels
- Some noise levels show ≥100% retention
- Need to verify which noise levels achieve 101%+

**Approach**:
- Results: Report empirical observation ("M2 maintained or exceeded baseline performance at noise levels σ ∈ {X, Y, Z}")
- Discussion: "This phenomenon is consistent with stochastic resonance [citations], where mild noise improves signal detection"

**Citations to add**:
- Gammaitoni, L., Hänggi, P., Jung, P., & Marchesoni, F. (1998). Stochastic resonance. Reviews of Modern Physics, 70(1), 223.
- McDonnell, M. D., & Abbott, D. (2009). What is stochastic resonance? Definitions, misconceptions, debates, and its relevance to biology. PLoS Computational Biology, 5(5), e1000348.

**ACTION**: Find exact noise levels where M2 achieves ≥100% retention

---

### 5. Camera-facing vs camera-away - NEEDS COMPUTATION
**DECISION**: Compute statistics and only include if significant

**Claim**: Anatomical patterns based on camera position affect robustness

**Joint groupings** (when robot walks left→right on screen):
- **Camera-away**: ankle_1, ankle_3 (left side joints)
- **Camera-facing**: ankle_2, ankle_4 (right side joints)

**What to compute**:
- Average retention for camera-away joints vs camera-facing joints
- Statistical significance (paired t-test with Bonferroni)
- Effect size (Cohen's d)

**ACTION**:
1. Define joint coordinate frame clearly
2. Group joints by camera position
3. Compute retention statistics
4. Only include in paper if statistically significant (p<0.05)

---

## COMPLETE STATISTICAL ANALYSIS TODO

### Experiment 1: Baseline Performance ✅
- [x] Mean ± std for all 4 models
- [x] 95% confidence intervals
- [x] Pairwise comparisons with Bonferroni correction
- [x] Cohen's d effect sizes

### Experiment 2: Sensor Noise Robustness ⏳
- [x] Performance at σ=0.10 (10× training noise)
- [x] Degradation percentages
- [ ] Find all noise levels where M2 achieves ≥100% retention
- [ ] Statistical significance of noise impact (all 12 noise levels)

### Experiment 3: Joint Failure Robustness ⏳
- [ ] Calculate retention percentages (baseline from Exp1)
- [ ] Mean retention across 8 joints per model
- [ ] M3 vs M4 comparison (key finding)
- [ ] Hip vs ankle statistical comparison
- [ ] Front vs rear leg comparison
- [ ] Camera-facing vs camera-away comparison

### Experiment 4: Combined Stress ⏳
- [ ] Find highest stress scenario (50% failure + σ=0.05)
- [ ] M3 vs M4 statistical test
- [ ] Effect size calculation
- [ ] Verify 25% underperformance claim

### Experiment 5: Per-Joint Deep Dive
- [ ] Velocity profiling statistics (if needed for Results)

### Experiment 7: Joint-Noise Interaction ⏳
- [ ] Worst-case scenario identification
- [ ] Retention heatmap statistics
- [ ] Test for additive vs synergistic effects

---

## WRITING STRATEGY - RESULTS SECTION

### Structure (Approved):
1. **IV.A Baseline Performance** (Figure 1)
2. **IV.B Sensor Noise Robustness** (Figure 2)
3. **IV.C Joint Failure Robustness** (Figure 4)
4. **IV.D Robustness Method Comparison** (Figure 3) - BUILD UP TO M4 FAILURE
5. **IV.E Joint-Noise Interaction Effects** (Figure 5)

### Writing Guidelines:
- ✅ **High statistical rigor**: All means, stds, p-values, effect sizes
- ✅ **Chronological + thematic**: Progress through stressors
- ✅ **Build narrative**: Show M4 potential → reveal failure
- ✅ **Pure facts only**: ZERO interpretation (save for Discussion)
- ✅ **Back all claims**: Data, statistics, or citations

---

## NEXT STEPS

1. ⏳ **Finish statistics script** - compute all missing values (~45 min)
2. ⏳ **Verify all claims** - check data supports every statement (~15 min)
3. ⏳ **Write Results section** - complete with all statistics (~60 min)
4. ✅ **Review and refine** - ensure academic rigor

**Total time estimate**: ~2 hours for complete, publication-ready Results section

---

## NOTES FROM STATISTICAL ANALYSIS

### Surprising Findings:
- M4 baseline (5.34m) is MUCH worse than M3 (7.96m) - dramatic interference
- M4 showed 213% retention at σ=0.10 - data anomaly, need to investigate
- M2 showed only 90.9% retention at σ=0.10, not the claimed 101%

### Data Quality Checks Needed:
- [ ] Why does M4 have such low baseline in Exp1 but different in Exp2?
- [ ] Verify retention calculations use correct baseline
- [ ] Check for data structure inconsistencies across experiments

---

*Last updated: 2025-10-22*
*Status: Statistical verification in progress - Option A (complete analysis)*
