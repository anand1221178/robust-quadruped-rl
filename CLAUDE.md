# CLAUDE.md - Project Memory & Context

## Project Overview
**Research Project**: Robust Quadruped RL with SR2L (Smooth Regularized Reinforcement Learning)
**Objective**: Implement SR2L algorithm for robust quadruped locomotion using PPO and RealAnt simulation

## Current Status (August 28, 2025)
- **Phase**: 3 (SR2L Implementation) - Fixed critical bug, retraining from scratch
- **Best Models**: ppo_target_walking_llsm451b (1.31 m/s baseline)
- **Latest Training**: ppo_sr2l_gentle_v5mqpx3e (FAILED - 0.11 m/s), ppo_dr_ef9vy61f (training - 0.31 m/s at 4.25M steps)

## Project Phases
1. ✅ **Phase 1**: Environment Setup - Migrate from Ant-v4 to RealAnt
2. ✅ **Phase 2**: Train baseline that actually walks
3. 🔄 **Phase 3**: Add SR2L to smooth fast walking (in progress)
4. ⏳ **Phase 4**: Domain Randomization Implementation
5. ⏳ **Phase 5**: Combined DR+SR2L Training
6. ⏳ **Final**: Comprehensive evaluation and comparison

## Key Achievements
- **Speed**: Achieved 0.9 m/s walking (target: 1.0+ m/s) vs original 0.084 m/s
- **Success Rate**: 81-89% success with target walking approach
- **Approach**: Goal-directed navigation (A-to-B) worked better than aggressive speed rewards
- **Problem Identified**: Fast walking is jerky/spazzy, needs SR2L smoothing

## Current Models
### Active Models (in experiments/)
1. **ppo_target_walking_llsm451b** - Best baseline (0.9 m/s, goal-directed)
2. **ppo_fast_walking_71n0huse** - Speed comparison model
3. **ppo_sr2l_gentle_dpmpni64** - COMPLETED gentle SR2L (15M steps, initialized from fast walker)
4. **ppo_sr2l_corrected_nppuk7mn** - COMPLETED corrected SR2L (ONLY joint sensor perturbations)

### Archived Models (in archive/experiments/)
- 23 failed/old experiments moved to archive during cleanup

## Technical Implementation
### SR2L Algorithm
- **Formula**: L_total = L_PPO + λ × L_smooth
- **L_smooth**: E[||π(s) - π(s + δ)||²] (policy sensitivity to state perturbations)
- **Current Config**: λ=0.001, warmup=2M steps, 15M total steps

### Environment Setup
- **Base**: RealAnt-v0 (29D observation space)
- **Wrapper**: TargetWalkingWrapper (goal-directed navigation, 5m targets)
- **Normalization**: VecNormalize for observations and rewards

### Evaluation Metrics
1. **Speed**: Average velocity (m/s)
2. **Smoothness**: Action smoothness score = 1/(1 + mean(action_changes))
3. **Robustness**: Success rate under sensor noise (0%, 5%, 10% noise levels)
4. **Stability**: Angular velocity < 2.0 rad/s

## Recent Major Changes

### August 28, 2025 - CRITICAL SR2L BUG FOUND AND FIXED
- **Major Discovery**: Found critical gradient flow bug in SR2L implementation
- **Bug Location**: `src/agents/ppo_sr2l.py` line 62
  - Original actions computed with `torch.no_grad()` - detached from computational graph
  - SR2L loss couldn't properly backpropagate through original actions
  - Caused one-sided regularization and policy instability
- **Fix Applied**: 
  - Removed `torch.no_grad()` wrapper from original action computation
  - Added `noise.detach()` to prevent gradients through random perturbations
  - Now both original and perturbed actions have proper gradients for bidirectional regularization
- **New Training Strategy**: `configs/experiments/ppo_sr2l_scratch.yaml`
  - Train from SCRATCH (no pretrained model conflicts)
  - Very gentle parameters: λ=0.0005, std=0.005, max_perturb=0.02
  - 1M step warmup before SR2L kicks in, 20M total training
  - Still uses target walking for proper goal-directed locomotion
- **Why All Previous SR2L Failed**: 
  - Gradient flow bug prevented proper regularization
  - Pretrained model weights conflicted with SR2L objectives
  - Too aggressive regularization parameters
  - All attempts (gentle, aggressive, motor, observation) failed due to same bug

### August 23, 2025 - TRUE SR2L Motor Perturbation FAILURE
- **Final Fix Applied**: Now perturbing ACTIONS (motor outputs) instead of observations
- **Implementation**: Simulates 5-10% motor degradation as per research proposal
- **Results**: STILL CATASTROPHIC FAILURE
  - **Speed**: 0.25 ± 0.01 m/s (only 19% of baseline's 1.31 m/s) 
  - **Consistency**: Very consistent failure (low std dev)
- **Conclusion**: SR2L fundamentally breaks target walking, regardless of implementation
- **All 4 SR2L attempts failed**: Aggressive, Gentle, Observation-perturbed, Action-perturbed

### August 23, 2025 - Final Gentle SR2L Results
- **Velocity (Corrected Test)**: 
  - **Baseline**: 1.31 ± 0.05 m/s (very consistent, 131% of target)
  - **Gentle SR2L**: 1.24 ± 0.16 m/s (maintains 95% speed, more variable, 124% of target)
- **Smoothness**: 30% improvement in mean action changes (1.014 → 0.711)
- **Robustness Issues**: Still shows 8-30% worse success rates under noise
- **Performance**: Better than aggressive SR2L but still worse than baseline
- **Status**: Gentle SR2L partially successful - good speed + smoothness, poor robustness

## Recent Major Changes

### August 23, 2025 - Directory Cleanup
- Cleaned up project directory, moved 23 old experiments to archive
- Kept only 2 essential models and 4 working configs
- Freed ~1.2GB space, organized archive structure
- Preserved all work in organized archive/ folders
- Added archive/ to .gitignore to prevent large file uploads

### August 23, 2025 - SR2L Fix Attempt
- **Problem**: Original SR2L failed catastrophically (81% → 26% success)
- **Solution**: Created ppo_sr2l_gentle with:
  - Initialization from successful target walker
  - Much gentler regularization (λ=0.001 vs 0.005)
  - Longer warmup period (2M steps)
  - 15M total training steps
- **Status**: Currently training

### Recent Training Results
- **Target Walking**: 0.896 ± 0.029 m/s (very consistent)
- **Fast Walking**: 0.748 ± 0.373 m/s (more variable)
- **Winner**: Target walking approach for consistent speed

## Key Files & Scripts
### Essential Scripts (scripts/)
- `debug_velocity.py` - Test walking speed of models
- `evaluate_sr2l.py` - Compare PPO vs SR2L with noise testing
- `record_video.py` - Create clean videos of models
- `test_trained_model.py` - Basic model testing

### Essential Configs (configs/experiments/)
- `ppo_target_walking.yaml` - Best baseline config
- `ppo_fast_walking.yaml` - Speed-focused config  
- `ppo_sr2l_gentle.yaml` - Gentle SR2L config
- `ppo_baseline.yaml` - Reference config

### Core Implementation (src/)
- `train.py` - Main training script with config system
- `agents/ppo_sr2l.py` - Custom PPO+SR2L implementation
- `envs/target_walking_wrapper.py` - Goal-directed navigation
- `envs/success_reward_wrapper.py` - Speed-focused rewards

## SR2L FUNDAMENTAL FIX - ACTION PERTURBATION
**CRITICAL MISUNDERSTANDING FIXED**: SR2L should perturb ACTIONS (motor outputs), not observations!

**Research Proposal Intent**:
- SR2L: Handle **degrading/failing motors** (motors not producing expected torque)
- Domain Randomization: Handle **locked/failed joints** (complete joint failure)

**What We Were Doing Wrong**:
- ❌ Perturbing sensor observations (confusing what robot sees)
- ❌ Breaking the navigation and balance systems

**Correct Implementation (NOW FIXED)**:
- ✅ Perturb ACTIONS (motor commands) to simulate worn/degrading motors
- ✅ Robot learns policies robust to imperfect motor execution
- ✅ Simulates realistic motor wear: 5-10% torque degradation

## Research Proposal: ABLATION STUDY
**Goal**: Compare robustness approaches for quadruped locomotion

**Ablation Components**:
1. **PPO (Baseline)**: ppo_target_walking_llsm451b - 1.31 m/s ✅
2. **PPO + SR2L**: ppo_sr2l_gentle_dpmpni64 - 1.24 m/s (best SR2L) ✅
3. **PPO + DR**: To be implemented (Phase 4) ⏳
4. **PPO + SR2L + DR**: To be implemented (Phase 5) ⏳

**CRITICAL RESEARCH PROPOSAL ANALYSIS**:
**SR2L Purpose (from proposal)**: Handle **sensor noise** and **noisy proprioceptive signals**
- This means **observation perturbation** was CORRECT all along!
- Motor perturbation was misinterpretation of proposal intent

**Research Proposal Scope**:
- **SR2L**: Sensor noise robustness (observation perturbations)
- **Domain Randomization**: Actuator failures (joint lock/dropout)  
- **Target Walking**: Valid training method (forces real locomotion vs lazy strategies)

**Best SR2L Model**: Gentle SR2L (1.24 m/s, 95% baseline speed) - RETRAINING
- Uses observation perturbations (correct for sensor noise per proposal)
- Config and implementation verified for retraining

## Supervisor Communication
- Evaluation metrics are comprehensive (speed, smoothness, robustness)
- Have quantitative ways to measure smoothness improvements
- Initial results show SR2L potential but need to fix performance degradation

---
*This file is updated after every significant change or conversation to maintain project context and memory.*