# W&B Monitoring Guide - SmoothTargetWrapper Experiments

**Project**: robust-quadruped-rl  
**Phase 1**: Baseline + SR2L with goal-directed locomotion  
**Started**: September 7, 2025

## 🎯 Current Experiments

### Phase 1 - ACTIVE (September 7, 2025)
1. **ppo_smooth_baseline** (30M steps, ~24h)
   - Job: `rohl32fn` 
   - Goal: Perfect A-to-B locomotion baseline
   
2. **ppo_smooth_sr2l** (40M steps, ~32h)  
   - Job: `ibfwtp9t`
   - Goal: Sensor noise robustness + A-to-B locomotion

### Phase 2 - PENDING  
3. **ppo_smooth_persistent_dr** (50M steps)
4. **ppo_smooth_permanent_dr** (60M steps)

## 📊 Key W&B Metrics to Monitor

### 🔥 MOST CRITICAL - Episode Reward (`rollout/ep_rew_mean`)
- **Current**: ~3000 ✅ (both models)
- **Should**: Increase steadily over training
- **Milestones**:
  - Week 1: Reach 4000+ 
  - Week 2: Reach 5000+ (multiple targets per episode)
- **🚨 RED FLAG**: Plateau below 3500 or decreasing

### 🤖 Robot Health - Episode Length (`rollout/ep_len_mean`)  
- **Current**: 1000 (max episode) ✅
- **Should**: Stay at 1000 (robot not falling)
- **🚨 RED FLAG**: Drops below 800 (frequent falling)

### 🧠 Learning Progress - Explained Variance (`train/explained_variance`)
- **Current**: Negative (normal early training) ✅
- **Should**: Increase toward 0.5-0.8
- **Goal**: Positive and stable (value function learning)
- **🚨 RED FLAG**: Stays negative after 5M steps

### 📉 Value Function - Value Loss (`train/value_loss`)
- **Current**: ~0.3-0.7 ✅
- **Should**: Decrease over time  
- **Goal**: Stabilize around 0.1-0.3
- **🚨 RED FLAG**: Explodes above 2.0

### ⚖️ Policy Stability - Policy Gradient Loss (`train/policy_gradient_loss`)
- **Current**: Small negative values ✅
- **Should**: Oscillate around 0, decreasing magnitude
- **🚨 RED FLAG**: Large positive/negative values (>0.1)

### 🎲 Exploration - Entropy Loss (`train/entropy_loss`)
- **Current**: ~-11.3 ✅ 
- **Should**: Gradually decrease (less random actions)
- **🚨 RED FLAG**: Drops too fast (premature convergence)

## 🎯 Success Timeline Expectations

### **Hours 1-6: Startup Phase**
- Episode reward: 2500 → 3500
- Explained variance: -0.5 → -0.1  
- Value loss: 0.7 → 0.5
- **Success**: Steady upward trends

### **Hours 6-24: Learning Phase** 
- Episode reward: 3500 → 4500
- Explained variance: -0.1 → +0.3
- Value loss: 0.5 → 0.3
- **Success**: First target completions visible

### **Hours 24-48: Mastery Phase**
- Episode reward: 4500 → 5500+
- Explained variance: 0.3 → 0.6+
- Value loss: 0.3 → 0.2
- **Success**: Multiple targets per episode

### **Days 2+: Optimization Phase**
- Episode reward: 5500+ and stable
- All metrics stable and healthy
- **Success**: Consistent A-to-B locomotion

## 🔥 SR2L-Specific Metrics (ppo_smooth_sr2l only)

### **Smoothness Loss** (if available)
- **Should**: Decrease over time (actions more consistent)
- **Goal**: Stable low values
- **🚨 RED FLAG**: Increasing or oscillating wildly

### **KL Divergence** (`train/approx_kl`)
- **Should**: Stay moderate (0.01-0.05)
- **🚨 RED FLAG**: Too high (>0.1) = over-regularization

## 🚨 CRITICAL Red Flags - STOP TRAINING IF:

1. **NaN Values**: Any metric shows NaN
2. **Reward Collapse**: Episode reward drops below 1000
3. **Episode Length Crash**: Below 300 consistently  
4. **Value Loss Explosion**: Above 5.0
5. **No Learning**: No improvement after 5M steps

## 📈 Custom Metrics to Watch (If Available)

### **Goal-Directed Behavior**:
- `distance_to_target`: Should decrease within episodes
- `targets_reached`: Should increase per episode over time
- `progress`: Should be consistently positive
- `velocity`: Should stabilize around target (~0.3 m/s)

### **Smoothness (SR2L)**:
- `smoothness_score`: Should increase over time
- `action_consistency`: Should improve

## ✅ Daily Checklist

### **Morning Check** (every 24h):
- [ ] Both jobs still running (no crashes/timeouts)
- [ ] Episode rewards trending upward
- [ ] No red flag metrics
- [ ] GPU utilization healthy

### **Evening Check** (every 12h):
- [ ] Learning curves smooth (not oscillating wildly)
- [ ] Value loss decreasing
- [ ] Explained variance improving

## 🎯 Phase 1 Success Criteria

**Baseline Success** (ppo_smooth_baseline):
- Episode reward >5000 consistently
- Smooth locomotion to targets
- No falling/crashing
- Ready for Phase 2 fine-tuning

**SR2L Success** (ppo_smooth_sr2l):  
- Episode reward >4500 (slightly lower OK due to regularization)
- Smooth actions + goal reaching
- No NaN crashes (tanh working)
- Robustness to sensor noise demonstrated

## 📞 When to Intervene

### **STOP and Debug** if:
- No improvement after 48 hours
- Any critical red flags triggered
- Jobs crash repeatedly

### **Proceed to Phase 2** if:
- Both models show consistent learning
- Success criteria met
- Ready for DR experiments

---

**Last Updated**: September 7, 2025  
**Next Review**: Check every 12 hours during Phase 1