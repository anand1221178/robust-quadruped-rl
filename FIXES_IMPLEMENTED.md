# SR2L and DR Fixes Implemented (September 9, 2025)

## Overview
Based on supervisor feedback, we've identified and fixed fundamental issues with both SR2L and standard DR approaches.

## 1. SR2L Fix - Tanh Activation

### Problem Identified
- Using ReLU activation with bounded action spaces causes unbounded outputs
- This leads to extreme torques → physics instability → NaN values in joint velocities/positions
- RealAnt uses Box action space (bounded to [-1, 1])

### Solution Implemented
**File**: `configs/experiments/ppo_sr2l_tanh.yaml`
- Changed activation from `relu` to `tanh`
- Tanh naturally bounds outputs to [-1, 1] matching action space
- Following Schulman et al. 2017 recommendation for bounded action spaces
- Training from scratch to avoid ReLU pretrained model conflicts

### Expected Outcome
- Should eliminate NaN issues during training
- If still fails, provides strong negative result about SR2L incompatibility with locomotion
- Important research contribution either way

## 2. Domain Randomization Fix - Persistent Failures

### Problem Identified  
- Current DR uses single-timestep failures (unrealistic!)
- Joint fails for 1 step then magically recovers
- Real hardware failures persist for extended periods
- This explains why DR only moderately improves robustness

### Solution Implemented
**Files**: 
- `src/envs/persistent_dr_wrapper.py` - New wrapper with realistic failure durations
- `configs/experiments/ppo_persistent_dr.yaml` - Training configuration

### Failure Duration Categories
1. **Short failures** (50-200 steps): ~2.5-10 seconds - component stress
2. **Medium failures** (200-1000 steps): ~10-50 seconds - temporary damage  
3. **Long failures** (entire episode): Permanent for that trial - catastrophic failure

### Key Improvements
- Failures persist realistically (not single timesteps)
- Forces robot to learn true adaptation strategies
- More closely simulates real-world hardware issues
- Progressive curriculum like successful DR v2

## 3. Comparison of DR Approaches

| Approach | Failure Duration | Realism | Expected Learning |
|----------|-----------------|---------|-------------------|
| Standard DR (current) | 1 timestep | Unrealistic | Momentary recovery |
| Persistent DR (new) | 50-1000+ steps | Realistic | True adaptation |
| Permanent DR (training) | Forever | Extreme test | Complete compensation |

## 4. Training Plan

### Phase 1: SR2L with Tanh
```bash
sbatch scripts/train_ppo_cluster.sh ppo_sr2l_tanh
```
- 30M steps with 8M warmup
- Monitor for NaN issues
- Document as negative result if fails

### Phase 2: Persistent DR
```bash
sbatch scripts/train_ppo_cluster.sh ppo_persistent_dr
```
- 30M steps matching DR v2 timeline
- Compare to standard DR and permanent DR
- Should outperform both at moderate failure scenarios

### Phase 3: Evaluation
- Permanent DR should complete ~Sept 6 (already training)
- Compare all approaches:
  - Baseline: 0.214 m/s (no robustness)
  - SR2L + Tanh: Sensor noise robustness (if it works)
  - Standard DR: 0.178 m/s (short-term failures)
  - Persistent DR: Medium-term failures (expected best)
  - Permanent DR: Extreme failures

## 5. Research Narrative

The fixes address fundamental flaws:
- **SR2L**: Missing tanh for bounded actions → NaN crashes
- **Standard DR**: Unrealistic single-step failures → poor adaptation

With these fixes:
- SR2L handles sensor noise (if tanh works)
- Persistent DR handles realistic hardware failures
- Permanent DR handles catastrophic failures

This creates a complete robustness spectrum for the ablation study.

## 6. Next Steps

1. Launch SR2L with tanh training on cluster
2. Launch persistent DR training on cluster
3. Monitor permanent DR completion (~Sept 6)
4. Create comprehensive evaluation comparing all approaches
5. Document findings (including SR2L as negative result if it fails)

## Key Insight for Supervisor Response

"The single-timestep failures in standard DR are fundamentally unrealistic - real hardware doesn't fail for 50ms then recover. Our new persistent DR with 50-1000+ timestep failures forces the robot to learn true adaptive strategies, not just momentary recovery reflexes."