# RealANT Deployment: Two Approaches Explained

This guide compares the two methods for deploying your trained policies to the physical RealANT robot.

---

## Quick Comparison

| Feature | Simulation Mirroring | Direct Policy Deployment |
|---------|---------------------|------------------------|
| **Script** | `simulation_mirroring.py` | `deploy_robust_policy.py` |
| **Visualization** | ✅ MuJoCo GUI shows what policy "thinks" | ❌ No visualization |
| **Sensors Used** | Simulated (policy never sees real robot) | Real (AR tags + servo encoders) |
| **Debugging** | ✅ Easy - compare sim vs reality side-by-side | ⚠️ Harder - blind to policy's internal state |
| **Hardware Required** | Servos only | Servos + Camera + AR tags + Calibration |
| **Sim-to-Real Gap** | ⚠️ Large (policy unaware of real dynamics) | ✅ Smaller (policy adapts to real observations) |
| **Setup Complexity** | ⭐⭐ Medium | ⭐⭐⭐⭐ High |
| **Best For** | Initial testing, visualization, debugging | Final evaluation, publication results |

---

## Approach 1: Simulation Mirroring (Easier, Better for Debugging)

### How It Works

```
┌──────────────────────────────┐
│  Your Laptop                 │
│                              │
│  ┌────────────────────────┐  │
│  │ MuJoCo Simulation      │  │
│  │ - Runs @ 50Hz          │  │
│  │ - Policy reads SIM obs │  │
│  │ - Shows GUI            │  │
│  │ - Computes actions     │  │
│  └───────┬────────────────┘  │
│          │                   │
│          │ Extract joint     │
│          │ positions         │
│          ▼                   │
│  ┌────────────────────────┐  │
│  │ ZeroMQ Bridge          │  │
│  │ - Converts to Dynamixel│  │
│  │ - Safety limits        │  │
│  └───────┬────────────────┘  │
└──────────┼───────────────────┘
           │ USB
           ▼
┌──────────────────────────────┐
│  Physical RealANT Robot      │
│  - Blindly mirrors sim       │
│  - No sensor feedback to AI  │
└──────────────────────────────┘
```

### What You See

**MuJoCo Window:**
- Simulated robot walking forward
- Policy thinks everything is perfect
- Clean, smooth motion

**Physical Robot:**
- Tries to mimic simulated joint angles
- May behave differently due to:
  - Floor friction
  - Servo backlash
  - Weight distribution
  - Timing delays

### Setup Steps

1. **Minimal Hardware Setup** (30 min)
   - Assemble robot
   - Connect servos to OpenCM board
   - Plug USB into laptop
   - Upload firmware (`ant11_cmd_dxl` or `ant14_cmd_dxl_nano33iot`)

2. **Test Servo Communication** (5 min)
   ```bash
   cd deployment/realant_hardware/realant
   python ant_server.py --port /dev/ttyUSB0
   ```
   Should detect all 8 servos.

3. **Run Simulation Mirroring** (2 min)
   ```bash
   cd deployment/realant_hardware

   # Terminal 1: Start servo controller
   python realant/ant_server.py --port /dev/ttyUSB0

   # Terminal 2: Run mirroring with M3 model
   python simulation_mirroring.py \
       --model ../../experiments/M3_dr_seed42/best_model/best_model.zip \
       --vec-normalize ../../experiments/M3_dr_seed42/vec_normalize.pkl \
       --render \
       --episodes 3
   ```

### When to Use This Approach

✅ **Use simulation mirroring when:**
- You want to **quickly test** if the robot hardware works
- You want to **see what the policy is doing** (MuJoCo visualization)
- You're **debugging** why real robot behaves differently than simulation
- You want to **record side-by-side videos** (sim vs reality)
- You're **early in deployment** and just getting hardware working
- You don't have time to set up camera/AR tags yet

❌ **Don't use for:**
- Final publication results (sim-to-real gap will be large)
- Quantitative performance metrics
- Demonstrating robustness (policy doesn't see real perturbations)

### Expected Results

**Simulation** (what you'll see in MuJoCo):
- Perfect locomotion
- 7-8m in 10 seconds
- Smooth, stable gait

**Physical Robot**:
- May walk ~50-70% as well as simulation
- Will likely veer off course (open-loop control)
- May fall if floor friction very different from sim
- Still useful for debugging!

---

## Approach 2: Direct Policy Deployment (Harder, Better Results)

### How It Works

```
┌──────────────────────────────────────┐
│  Physical World                      │
│                                      │
│  📷 Camera (overhead)                │
│     ↓ AR tag tracking                │
│  ┌──────────────────┐                │
│  │ 🤖 RealANT Robot │                │
│  │ - Joint encoders │                │
│  │ - Moves based on │                │
│  │   policy output  │                │
│  └────┬─────────────┘                │
│       │ Servo positions               │
└───────┼──────────────────────────────┘
        │
        │ ZeroMQ (real observations)
        ▼
┌──────────────────────────────────────┐
│  Your Laptop                         │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ Trained Policy                 │  │
│  │ - Receives REAL observations   │  │
│  │   (AR pose + joint encoders)   │  │
│  │ - VecNormalize processes them  │  │
│  │ - Computes action @ 20Hz       │  │
│  └────────┬───────────────────────┘  │
│           │                          │
│           │ Actions                  │
│           ▼                          │
│  ┌────────────────────────────────┐  │
│  │ ZeroMQ Action Publisher        │  │
│  └────────┬───────────────────────┘  │
└───────────┼──────────────────────────┘
            │ USB
            ▼
         Servos
```

### What You See

**No visualization** - but the policy is experiencing the REAL robot:
- Real sensor noise
- Real servo delays
- Real floor friction
- Real perturbations

This is **closed-loop control** - policy adapts to what it senses.

### Setup Steps

1. **Full Hardware Setup** (2-3 hours first time)
   - Assemble robot ✅
   - Connect servos ✅
   - **Mount camera 1-1.5m overhead** 📷
   - **Print and attach AR tags** (robot + floor)
   - **Set up lighting** (2-3 LED floodlights)
   - **Calibrate camera** (chessboard calibration)
   - **Test AR tag tracking**

2. **Test AR Tag Tracking** (15 min)
   ```bash
   cd deployment/realant_hardware/realant

   # Calibrate camera first
   python capture.py  # Capture chessboard images
   python calibrate_camera.py  # Generate cam_calib.pkl

   # Test AR tracking
   python showaruco_board.py --camera 0 --calibration cam_calib.pkl
   ```
   Should show live camera feed with both tags detected.

3. **Run Direct Policy Deployment** (5 min)
   ```bash
   cd deployment/realant_hardware

   # Terminal 1: AR tag tracker
   python realant/showaruco_board.py \
       --camera 0 \
       --calibration realant/cam_calib.pkl \
       --port 5555

   # Terminal 2: Servo controller
   python realant/ant_server.py \
       --port /dev/ttyUSB0 \
       --pub-port 5557 \
       --sub-port 5556

   # Terminal 3: Deploy policy (reads real sensors!)
   python deploy_robust_policy.py \
       --model ../../experiments/M3_dr_seed42/best_model/best_model.zip \
       --vec-normalize ../../experiments/M3_dr_seed42/vec_normalize.pkl \
       --episodes 5
   ```

### When to Use This Approach

✅ **Use direct policy deployment when:**
- You want **publication-quality results**
- You want to **demonstrate true robustness** (policy handles real noise)
- You want **quantitative metrics** for paper
- You want to **compare sim vs real** performance accurately
- Your robot is **fully set up** with camera/AR tags

❌ **Don't use when:**
- You're just getting started (use mirroring first)
- Camera/AR setup not ready yet
- You want to debug what policy is "thinking" (can't see internal state)

### Expected Results

**M3 (Domain Randomization) Model:**
- Should achieve **6-7m in 10 seconds**
- ~25% degradation from simulation (expected)
- Robust to:
  - AR tag noise
  - Servo lag
  - Floor friction variation
- Should **NOT fall** (trained for robustness!)

---

## Recommendation: Phased Approach

### Phase 1: Quick Hardware Test (Day 1)
**Use Simulation Mirroring**
- Goal: Verify servos work, robot can move
- Setup time: ~1 hour
- Outcome: You'll see robot attempt to walk, but performance may be poor

### Phase 2: Full Deployment Setup (Day 2-3)
**Set Up Camera & AR Tags**
- Goal: Get direct policy deployment working
- Setup time: 2-3 hours
- Outcome: Camera calibrated, AR tracking working

### Phase 3: Real Results (Day 3-4)
**Use Direct Policy Deployment**
- Goal: Collect publication-quality data
- Setup time: 5 min (already done from Phase 2)
- Outcome: Real performance metrics for M1, M2, M3, M4

### Phase 4: Debugging (If Needed)
**Switch Between Both**
- Run mirroring to see what policy "should" do
- Run direct deployment to see what actually happens
- Compare side-by-side to identify sim-to-real gaps

---

## Troubleshooting: Which Approach to Debug With?

### Problem: "Robot doesn't walk at all"

**Debug with:** Simulation Mirroring
1. Run `simulation_mirroring.py` with `--render`
2. Watch MuJoCo window - does simulated robot walk?
   - **Yes** → Problem is sim-to-real transfer (mechanical issue)
   - **No** → Problem is model/policy (check model path, VecNormalize)

### Problem: "Robot walks but falls over"

**Debug with:** Simulation Mirroring + Direct Deployment
1. Run mirroring - does simulated robot fall?
   - **No** → Sim-to-real gap (e.g., weight distribution, friction)
   - **Yes** → Policy issue (may need different model)
2. Try different model (M2, M3) - M3 should be most robust

### Problem: "Robot veers off course"

**Debug with:** Check which approach you're using
- **Mirroring** → Expected (open-loop, no feedback)
- **Direct** → AR tag tracking issue or floor friction

### Problem: "Performance much worse than simulation"

**Debug with:** Both approaches
1. Run mirroring first - sets upper bound on performance
2. Then run direct - should be BETTER than mirroring (closed-loop)
3. If direct is WORSE than mirroring → sensor issue (AR tags, servos)

---

## Summary Table: Which Script to Run?

| Your Goal | Script to Use | Setup Required |
|-----------|--------------|----------------|
| "Does my robot hardware work?" | `simulation_mirroring.py` | Servos only |
| "What does my policy think it's doing?" | `simulation_mirroring.py --render` | Servos only |
| "Side-by-side sim vs reality video" | `simulation_mirroring.py` | Servos only |
| "Quick demo for advisor" | `simulation_mirroring.py` | Servos only |
| "Publication performance metrics" | `deploy_robust_policy.py` | Full (camera + AR tags) |
| "Demonstrate robustness claims" | `deploy_robust_policy.py` | Full |
| "Compare M1 vs M2 vs M3 vs M4" | `deploy_robust_policy.py` | Full |
| "Why is sim-to-real gap so large?" | Both (compare outputs) | Full |

---

## FAQ

**Q: Which approach did the previous student use?**
A: Likely simulation mirroring - it's faster to set up and good for initial testing.

**Q: Which should I use for my paper?**
A: Direct policy deployment (`deploy_robust_policy.py`) - it demonstrates true sim-to-real transfer with real sensor feedback.

**Q: Can I use both?**
A: Yes! Use mirroring first to verify hardware, then switch to direct deployment for results.

**Q: Why might mirroring perform poorly?**
A: The policy never sees real sensor noise, so it can't adapt. It's open-loop control - any small error accumulates.

**Q: Will my M3 (DR) model work better with direct deployment?**
A: YES! M3 was trained for joint failures and sensor noise - direct deployment lets it use that training. Mirroring doesn't benefit from robustness training.

**Q: How much worse will real robot perform vs simulation?**
A:
- **Mirroring:** 30-50% worse (large sim-to-real gap)
- **Direct Deployment:** 20-30% worse (policy adapts to real sensors)

---

## Next Steps

1. ✅ You now have both scripts available
2. ⏳ Choose your starting approach based on your timeline:
   - **Short on time?** Start with simulation mirroring
   - **Want best results?** Go straight to direct deployment
3. 📖 Follow the relevant guide:
   - **Mirroring:** This document (Approach 1 section)
   - **Direct:** `DEPLOYMENT_GUIDE.md` (comprehensive camera/AR setup)

**Good luck!** 🚀
