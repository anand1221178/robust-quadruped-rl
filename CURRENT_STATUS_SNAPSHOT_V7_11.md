# CURRENT STATUS SNAPSHOT - SEPTEMBER 28, 2025 (V7.11 ROTATION MASTERY)

**🎯 CURRENT FOCUS**: V7.11 Rotation Mastery training to solve ankle_4 problem through targeted rotation rewards

**PREVIOUS CHAMPION**: V7.7E Ultra Speed (0.539 m/s baseline, 12.7% ankle_4 retention with delayed locking)

---

## 🚀 **V7.11 ROTATION MASTERY - LAUNCHING SEPTEMBER 28, 2025**

### **The Ankle_4 Challenge**:
- **Problem**: Ankle_4 (rear-right, camera-facing) consistently performs worst (~5-12% retention)
- **Discovery**: Ankle_1 succeeds through lift-and-rotate strategy
- **Solution**: Teach ankle_4 the same rotation recovery strategy via targeted rewards

### **V7.11 Implementation**:
```python
# RotationMasteryWrapper - NaN-safe multiplicative rewards
- 1.5x reward for rotation when ankle_4 locked
- 2.0x reward for forward movement after rotation
- 2.5x hard cap to prevent NaN
- 70% of failures target ankle_4 for focused learning
```

### **Key Components**:
1. **RotationMasteryWrapper** (`src/envs/rotation_mastery_wrapper.py`)
   - Detects ankle_4 failure and rewards rotation behavior
   - NaN-safe multiplicative rewards prevent training crashes

2. **Weighted Joint Sampling** (enhanced CurriculumDRWrapper)
   - 70% ankle_4, 10% ankle_1, 20% others
   - Learns from ankle_1's successful strategy

3. **Fine-tuning from V7.7E**
   - Preserves 0.539 m/s baseline performance
   - 12M steps with ultra-low LR (5e-5)

### **Expected Outcomes**:
- **Target**: 25-35% ankle_4 retention (up from 12.7%)
- **Baseline**: Maintain 0.50+ m/s
- **Training Time**: 8-10 hours on cluster

---

## 📊 **JOURNEY FROM V7.6 TO V7.11**

### **V7.6C (Sept 22)**: BackwardPenaltyWrapper Breakthrough
- **Achievement**: Eliminated backward walking with 5x penalty
- **Performance**: 0.169 m/s baseline
- **Status**: Solved backward pathology, foundation for V7.7

### **V7.7 Series (Sept 23)**: Multi-tier Speed Bonuses
- **Launch**: 6 models trained in parallel
- **V7.7E Winner**: Ultra speed mode with multi-tier bonuses
  - 0.175 m/s immediate evaluation
  - 0.539 m/s with delayed locking (2-second delay)
- **Key**: All used CurriculumDRWrapper (avoided buggy ACDR)

### **V7.8 Series (Sept 23-24)**: Refinements
- **Built on V7.7E**: Various ankle_4 targeting approaches
- **Best**: V7.8A Ankle Specialist (3.1% ankle_4 improvement)
- **Learning**: Targeted training helps but not enough

### **V7.9 Series (Sept 24)**: Extended Episodes
- **V7.9A/B**: 2500-step episodes for rotation completion
- **V7.9C Failure**: 71% ankle_4 training caused overspecialization
- **Discovery**: Episode length matters for rotation strategies

### **V7.10 Series (Sept 27)**: Symmetric Training Attempt
- **V7.10C/D**: Attempted symmetric observation training
- **Result**: Catastrophic failure (0.021 m/s, robot stops randomly)
- **Discovery**: Symmetric training config was never implemented!
- **Learning**: Complex theoretical approaches can fail spectacularly

### **V7.11 (Sept 28)**: Rotation Mastery
- **Approach**: Surgical intervention - teach rotation explicitly
- **Foundation**: V7.7E pretrained (proven champion)
- **Innovation**: Reward rotation behavior when ankle_4 fails

---

## 🔍 **KEY TECHNICAL DISCOVERIES**

### **The ACDR "Drunk Robot" Bug** (Fixed but Avoided):
```python
# BUG in V7ACDRWrapper:
self.episode_return += reward  # Used normalized rewards!

# FIX in V7ACDRWrapperFixed:
self.episode_return += info.get('reward', reward)  # Raw rewards
```
**Impact**: ACDR curriculum was random. V7.7+ avoided it entirely.

### **Reward System Evolution**:
1. **V7.5 and earlier**: Various reward schemes
2. **V7.6C**: Added BackwardPenaltyWrapper (5x penalty)
3. **V7.7E**: Multi-tier speed bonuses (the magic formula)
4. **V7.11**: Adding rotation mastery multipliers

### **Ankle_4 Physics Insights**:
- **Position**: Rear-right, facing camera when walking left→right
- **Physics Issue**: Gets stuck at 0.699 radians when locked
- **Anatomical Challenge**: Perpendicular to movement + camera-facing
- **Solution Strategy**: Rotation to reposition failed joint

### **Symmetric Training Non-Implementation**:
- **Config existed** but no Python implementation
- **Result**: Models trained normally, not symmetrically
- **Learning**: Config ≠ Implementation

---

## 📁 **CURRENT MODEL HIERARCHY**

### **Champions**:
- **V7.7E Ultra Speed**: `done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/`
  - 0.539 m/s baseline (with delayed locking)
  - 12.7% ankle_4 retention
  - Multi-tier speed bonuses

### **Failed Experiments** (Valuable Lessons):
- **V7.10D Symmetric**: 0.021 m/s (catastrophic)
- **V7.9C Ankle4 Obsessed**: 0.001 m/s (overspecialization)

### **In Progress**:
- **V7.11 Rotation Mastery**: Ready to launch
  - Fine-tuning from V7.7E
  - Targeted ankle_4 rotation rewards

---

## 🎯 **IMMEDIATE NEXT STEPS**

1. **Launch V7.11**:
   ```bash
   sbatch scripts/train_ppo_cluster.sh v7_11_rotation_mastery
   ```

2. **Monitor Training**:
   - Watch W&B for stable learning
   - Check phase transitions at 8M steps
   - Verify no NaN issues

3. **Evaluation Protocol**:
   - Use championship script with 2-second delayed locking
   - Test all 8 individual joint failures
   - Focus on ankle_4 retention improvement

---

## 🔬 **RESEARCH INSIGHTS**

### **What Works**:
- ✅ Backward penalty (5x multiplier)
- ✅ Multi-tier speed bonuses
- ✅ Delayed joint locking for evaluation
- ✅ CurriculumDRWrapper (not ACDR)
- ✅ Fine-tuning from proven models

### **What Doesn't**:
- ❌ Symmetric observation flipping (destroys locomotion)
- ❌ Extreme specialization (V7.9C disaster)
- ❌ ACDR wrapper (drunk robot bug)
- ❌ Gentle penalties (2x insufficient)

### **Key Principles**:
1. **Raw Rewards**: Never clip/normalize in wrappers
2. **Multiplicative Bonuses**: Safer than additive
3. **Targeted Training**: Focus but don't overspecialize
4. **Build on Success**: Fine-tune from proven models

---

## ❌ **V7.11 CATASTROPHIC FAILURE - SEPTEMBER 29, 2025**

### **Training Complete - WORST PERFORMANCE YET**:
- **Baseline Speed**: 0.002 m/s (99.6% WORSE than V7.7E!)
- **Ankle_4 Retention**: 2.1% (WORSE than V7.7E's 12.7%)
- **Overall**: Complete locomotion failure - robot doesn't walk

### **Championship Results** (2-second delayed locking):
- Hip_1: 6.1% | Ankle_1: 22.3%
- Hip_2: 51.8% | Ankle_2: 21.5%
- Hip_3: 47.7% | Ankle_3: 3.0%
- Hip_4: 2.7% | **Ankle_4: 2.1%** (catastrophic)

### **Failure Analysis**:
1. **70% ankle_4 focus destroyed locomotion** (like V7.9C with 71%)
2. **Rotation rewards backfired** - encouraged spinning not walking
3. **Fine-tuning corrupted** despite V7.7E foundation
4. **Overspecialization disaster** - became expert at failing

## 🏆 **FINAL VERDICT**

**CHAMPION REMAINS**: V7.7E Ultra Speed (0.539 m/s, 12.7% ankle_4)

**Ankle_4 Practical Limit**: ~12-13% retention appears to be the maximum achievable

**Key Learning**: Aggressive targeted training (>50% focus) consistently destroys general locomotion

**Approaches That Failed**:
- ❌ V7.9C: 71% ankle_4 focus → 0.001 m/s
- ❌ V7.10: Symmetric training → 0.021 m/s
- ❌ V7.11: 70% ankle_4 rotation → 0.002 m/s

**Conclusion**: Accept ankle_4's anatomical limitations. V7.7E's balanced approach remains optimal.

---

*Final Update: September 29, 2025*
*V7.11 ROTATION MASTERY: Complete failure, worse than all previous attempts*
*V7.7E remains the undisputed champion* 🏆