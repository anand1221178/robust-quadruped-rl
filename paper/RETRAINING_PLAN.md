# Complete Retraining Plan - Fair 32M Comparison
**Created**: October 26, 2025
**Target Completion**: October 27-28, 2025 (before November 10 deadline)
**Total Compute**: 4 parallel jobs, ~14 hours wall-clock time

---

## 🎯 **OBJECTIVES**

1. ✅ **Eliminate training duration confound** - all models to 32M steps
2. ✅ **Improve DR curriculum** - remove dual failures, focus on single joint mastery
3. ✅ **Fix W&B logging** - proper phase transition tracking
4. ✅ **Potentially improve M3/M4 performance** - better curriculum design
5. ✅ **Enable clean comparison** - "All models trained to 32M steps"

---

## 📊 **TRAINING JOBS MATRIX**

| Model | Current | Target | Strategy | Compute Time | Priority |
|-------|---------|--------|----------|--------------|----------|
| **M1 (Baseline)** | 10M | 32M | ✅ Continue from checkpoint | ~10 hrs | 🔴 Critical |
| **M2 (SR2L)** | 20M | 32M | ✅ Continue from checkpoint | ~6 hrs | 🔴 Critical |
| **M3 (DR) NEW** | 32M old | 32M | 🔄 **Retrain from scratch** (new curriculum) | ~14 hrs | 🔥 Must do |
| **M4 (Combo) NEW** | 30M old | 32M | 🔄 **Retrain from scratch** (new curriculum) | ~14 hrs | 🔥 Must do |

**Parallel Execution**: All 4 jobs run simultaneously → **14 hours total wall time**

---

## 🔧 **IMPROVED DR CURRICULUM**

### **OLD Curriculum (Current M3/M4)**:
```yaml
Phase 1 (0-10M):
  - Clean training, no failures
  - Learn baseline locomotion

Phase 2 (10-20M):
  - 50% episodes with 1 random joint locked
  - 12M steps of single failure training

Phase 3 (20-32M):
  - 60% episodes with 1-2 random joints locked
  - 12M steps of DUAL failure training ❌ Complex!
```

**Total single-joint training**: 12M steps
**Dual-joint training**: 12M steps

### **NEW Curriculum (Proposed M3_v2/M4_v2)**:
```yaml
Phase 1 (0-10M):
  - Clean training, no failures
  - Learn baseline locomotion

Phase 2 (10-32M):
  - Single joint failures ONLY
  - Gradual ramp: 40% → 50% → 60% episode failure rate
  - 22M steps of focused single failure training ✅ Simpler!
  - NO dual failures - master one joint adaptation first
```

**Total single-joint training**: 22M steps (+83% more!)
**Dual-joint training**: 0M steps (removed)

### **Why This Should Improve Performance**:

1. **More mastery time**: 22M vs 12M steps on single failures
2. **Reduced complexity**: No dual failure confusion
3. **Better for M4**: Less gradient conflict between SR2L smoothness and DR adaptation
4. **Progressive difficulty**: Gradual ramp-up of failure rate
5. **Focused learning**: One task at a time

---

## 📝 **JOB 1: M1 Continuation (Critical)**

**Purpose**: Prove M1 convergence by extending training

### Configuration:
```yaml
experiment:
  name: ppo_baseline_32M_continuation
  description: "Continue M1 to 32M to prove convergence"

pretrained_model:
  path: done/ppo_baseline_ueqbjf2x/checkpoints/model_10000000_steps.zip
  vec_normalize: done/ppo_baseline_ueqbjf2x/checkpoints/model_vecnormalize_10000000_steps.pkl
  starting_timestep: 10_000_000

total_timesteps: 32_000_000  # Train for 22M more steps

# Same hyperparameters as original M1
ppo:
  learning_rate: 0.0003
  batch_size: 2048
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95

env:
  name: RealAntMujoco-v0
  use_success_reward: true

wandb:
  project: robust-quadruped-rl
  tags: [baseline, continuation, 32M, fair_comparison]
  notes: "M1 extended to 32M to eliminate training duration confound"
```

**Expected Outcome**: Performance stays ~11.20m ± 0.2m (proves convergence)

**Compute**: ~10 hours (22M steps / 661 steps/sec)

---

## 📝 **JOB 2: M2 Continuation**

**Purpose**: Extend SR2L to 32M for fair comparison

### Configuration:
```yaml
experiment:
  name: ppo_sr2l_32M_continuation
  description: "Continue M2 SR2L to 32M steps"

pretrained_model:
  path: done/ppo_sr2l_forward_m7gtjtpa/checkpoints/checkpoint_20000000_steps.zip
  vec_normalize: done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl
  starting_timestep: 20_000_000

total_timesteps: 32_000_000  # Train for 12M more steps

# Same SR2L hyperparameters
sr2l:
  lambda_smooth: 0.001
  perturbation_std: 0.01
  perturb_dims: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

ppo:
  learning_rate: 0.0003
  batch_size: 1536
  n_epochs: 8
  clip_range: 0.15

wandb:
  tags: [sr2l, continuation, 32M, fair_comparison]
```

**Expected Outcome**: Either stays ~8.75m or improves slightly

**Compute**: ~6 hours (12M steps / 661 steps/sec)

---

## 📝 **JOB 3: M3_v2 Complete Retrain (NEW CURRICULUM)**

**Purpose**: Retrain DR from scratch with improved single-failure-only curriculum

### Configuration:
```yaml
experiment:
  name: ppo_dr_v2_single_failures_32M
  description: "DR retrain - single failures only, no dual phase"

total_timesteps: 32_000_000

domain_randomization:
  enabled: true
  wrapper_type: CurriculumDRWrapper

  # IMPROVED CURRICULUM - Single Failures Only
  curriculum:
    phase_1:
      timestep_start: 0
      timestep_end: 10_000_000
      config:
        failure_prob: 0.0
        min_dropped_joints: 0
        max_dropped_joints: 0
      description: "Phase 1: Clean baseline learning"

    phase_2:
      timestep_start: 10_000_000
      timestep_end: 32_000_000
      config:
        failure_prob: 0.50  # 50% episodes with failure
        min_dropped_joints: 1  # Always exactly 1 joint
        max_dropped_joints: 1  # Never more than 1
        progressive_ramp: true  # Gradual 40→50→60% over 22M steps
      description: "Phase 2: Single joint failure mastery (22M steps)"

wandb:
  project: robust-quadruped-rl
  tags: [dr, v2, single_failures, curriculum_improved, 32M]
  notes: "NEW curriculum - removed dual failures, 22M steps single-joint training"

  # CUSTOM CALLBACK FOR PHASE LOGGING
  log_phase_transitions: true
  phase_markers:
    - timestep: 10_000_000
      label: "Phase 2 Start: Single Failures Begin"
```

**Expected Outcome**:
- Baseline: 7.5-8.5m (potentially better than old M3's 7.96m)
- Joint failure robustness: Improved due to more training time

**Compute**: ~14 hours (32M steps from scratch)

---

## 📝 **JOB 4: M4_v2 Complete Retrain (NEW CURRICULUM + SR2L)**

**Purpose**: Retrain SR2L+DR with improved curriculum - might reduce gradient conflict!

### Configuration:
```yaml
experiment:
  name: ppo_combo_v2_single_failures_32M
  description: "SR2L+DR retrain - improved curriculum, might fix collapse!"

total_timesteps: 32_000_000

# SR2L component (same as before)
sr2l:
  lambda_smooth: 0.001
  perturbation_std: 0.01
  perturb_dims: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

# NEW DR curriculum (same as M3_v2)
domain_randomization:
  enabled: true
  wrapper_type: CurriculumDRWrapper
  curriculum:
    phase_1:
      timestep_start: 0
      timestep_end: 10_000_000
      config:
        failure_prob: 0.0
        min_dropped_joints: 0
        max_dropped_joints: 0

    phase_2:
      timestep_start: 10_000_000
      timestep_end: 32_000_000
      config:
        failure_prob: 0.50
        min_dropped_joints: 1  # Single failures only!
        max_dropped_joints: 1
        progressive_ramp: true

wandb:
  tags: [combo, v2, single_failures, sr2l_plus_dr, 32M]
  notes: "Hypothesis: Simpler DR curriculum reduces gradient conflict with SR2L"
  log_phase_transitions: true
```

**Expected Outcome**:
- **Might avoid catastrophic collapse!** (simpler curriculum = less interference)
- Baseline: 6-8m (potentially better than old M4's 5.34m)
- **Key test**: Does simpler curriculum fix the gradient conflict?

**Compute**: ~14 hours (32M steps from scratch)

---

## 🎨 **W&B LOGGING IMPROVEMENTS**

### **Problem**: Current callbacks don't show phase transitions properly

### **Solution**: Custom callback for phase logging

**New file**: `src/callbacks/phase_logging_callback.py`

```python
from stable_baselines3.common.callbacks import BaseCallback
import wandb

class PhaseTransitionCallback(BaseCallback):
    """Logs phase transitions to W&B with visual markers"""

    def __init__(self, phase_config, verbose=0):
        super().__init__(verbose)
        self.phase_config = phase_config
        self.logged_phases = set()

    def _on_step(self) -> bool:
        """Check if we've entered a new phase"""
        current_timestep = self.num_timesteps

        for phase_name, phase_info in self.phase_config.items():
            phase_start = phase_info['timestep_start']

            # Log phase transition
            if (current_timestep >= phase_start and
                phase_name not in self.logged_phases):

                self.logged_phases.add(phase_name)

                # Log to W&B with custom event
                if wandb.run is not None:
                    wandb.log({
                        "phase_transition": phase_start,
                        "phase_name": phase_name,
                        "phase_description": phase_info.get('description', ''),
                        "failure_prob": phase_info['config'].get('failure_prob', 0),
                        "max_joints": phase_info['config'].get('max_dropped_joints', 0)
                    }, step=current_timestep)

                    # Add vertical line marker in W&B UI
                    wandb.run.summary[f"phase_{phase_name}_start"] = phase_start

                if self.verbose > 0:
                    print(f"\n{'='*60}")
                    print(f"🎯 PHASE TRANSITION @ {current_timestep:,} steps")
                    print(f"   Phase: {phase_name}")
                    print(f"   Description: {phase_info.get('description', '')}")
                    print(f"   Config: {phase_info['config']}")
                    print(f"{'='*60}\n")

        return True
```

**Usage in train.py**:
```python
from callbacks.phase_logging_callback import PhaseTransitionCallback

# Add to callbacks list
callbacks = [
    CheckpointCallback(...),
    PhaseTransitionCallback(
        phase_config=config.domain_randomization.curriculum,
        verbose=1
    )
]
```

**W&B Benefits**:
- ✅ Vertical lines at phase transitions in training curves
- ✅ Annotations showing curriculum changes
- ✅ Easy to correlate performance changes with phase shifts
- ✅ Summary stats showing exact phase start timesteps

---

## ⏱️ **TIMELINE & EXECUTION**

### **Day 1 (October 26 - Today)**:
- [x] ✅ Identify the plan
- [ ] Create 4 config files (2 continuations + 2 retrains)
- [ ] Update train.py to support checkpoint continuation
- [ ] Create PhaseTransitionCallback
- [ ] Submit all 4 jobs in parallel

### **Day 2 (October 27)**:
- [ ] Monitor training (~14 hours)
- [ ] Check W&B logs for proper phase transitions
- [ ] Verify no crashes/divergence

### **Day 3-4 (October 28-29)**:
- [ ] All jobs complete
- [ ] Run evaluation suite on new checkpoints
- [ ] Compare old M3 vs new M3_v2 (curriculum ablation!)
- [ ] Update paper with results

### **Day 5-14 (Oct 30 - Nov 10)**:
- [ ] Update paper text
- [ ] Address other supervisor comments
- [ ] Final revisions
- [ ] Submit paper

**Buffer**: 11 days for paper writing/revision (plenty of time!)

---

## 🎯 **EXPECTED OUTCOMES**

### **M1 @ 32M**:
- Performance: ~11.20m ± 0.2m (unchanged from 10M)
- **Proves**: M1 converged early, training duration wasn't the issue

### **M2 @ 32M**:
- Performance: ~8.5-9.0m (might improve slightly)
- **Outcome**: Either confirms 20M was enough, or shows small improvement

### **M3_v2 @ 32M (New Curriculum)**:
- Performance: **8.0-8.5m** (potentially better than old 7.96m!)
- Robustness: **Improved** (22M vs 12M single-failure training)
- **Key finding**: Simpler curriculum = better performance

### **M4_v2 @ 32M (New Curriculum + SR2L)**:
- Performance: **6.5-7.5m** (potentially MUCH better than old 5.34m!)
- **Critical test**: Does 14-20M collapse still happen?
- **If no collapse**: Proves dual failures caused gradient conflict!
- **If still collapse**: Confirms SR2L+DR fundamentally incompatible

---

## 📊 **ABLATION COMPARISONS ENABLED**

With old + new models, we can do extra ablations:

1. **Curriculum Complexity Ablation**:
   - M3_old (dual failures): 7.96m baseline, 47% retention
   - M3_v2 (single only): ?m baseline, ?% retention
   - **Tests**: Does simpler curriculum improve both baseline AND robustness?

2. **Gradient Conflict Resolution**:
   - M4_old (dual + SR2L): 5.34m, collapsed at 14-20M
   - M4_v2 (single + SR2L): ?m, collapse or not?
   - **Tests**: Was collapse from curriculum complexity?

3. **Training Duration Effects**:
   - M1 @ 10M vs 32M: Proves convergence
   - M2 @ 20M vs 32M: Tests if SR2L needed more training

---

## 🚀 **NEXT STEPS - READY TO EXECUTE?**

**I can create:**
1. ✅ 4 Hydra config files (2 continuations + 2 retrains)
2. ✅ Updated train.py with continuation support
3. ✅ PhaseTransitionCallback for W&B logging
4. ✅ 4 cluster submission scripts (sbatch)
5. ✅ Monitoring script to check job progress

**Do you want me to:**
- [ ] Generate all configs and scripts now?
- [ ] Or adjust the plan first (different timesteps, curriculum params, etc.)?

**Questions:**
1. Should we do 32M for all, or would 25M/30M be better?
2. For Phase 2 progressive ramp - linear 40→60% or step-wise?
3. Any other curriculum tweaks you want to try?

This plan gives you the **cleanest possible paper** AND might discover that simpler curriculum fixes M4! 🔥
