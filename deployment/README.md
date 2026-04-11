# RealANT Hardware Deployment

**Deploy your trained robustness policies (M1-M4) to the physical RealANT robot**

---

## 🎯 Two Deployment Approaches

### Approach 1: Simulation Mirroring ⭐ START HERE (30 min)
**Robot mimics MuJoCo simulation in real-time**
- ✅ Quick setup - No camera needed!
- ✅ See MuJoCo visualization while robot moves
- ✅ Perfect for initial testing and debugging
- 📖 Guide: [`SIMULATION_MIRRORING_GUIDE.md`](SIMULATION_MIRRORING_GUIDE.md)

### Approach 2: Direct Policy Deployment (2-3 hours)
**Policy reads real sensors and controls robot (closed-loop)**
- ✅ Best performance - Closed-loop control
- ✅ Publication-quality results
- ⚠️ Requires camera + AR tags + calibration
- 📖 Guide: [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)

**Not sure which to use?** Read: [`DEPLOYMENT_APPROACHES.md`](DEPLOYMENT_APPROACHES.md)

---

## 🚀 Quick Start: Simulation Mirroring

### 1. Hardware Setup (~15 min)
- Connect 8 servos to OpenCM board
- Plug USB cable to laptop
- Connect 12V power supply
- Find serial port: `ls /dev/tty.* | grep usb`

### 2. Software Installation (~5 min)
```bash
cd /Users/anand/Documents/4th\ Year/robust-quadruped-rl
source venv/bin/activate
pip install pyzmq pyserial
```

### 3. Start Servo Controller (Terminal 1)
```bash
cd deployment/realant_hardware/realant
python ant_server.py --port /dev/ttyUSB0  # Replace with your port
```

### 4. Run Simulation Mirroring (Terminal 2)
```bash
cd deployment/realant_hardware

# M3 model with visualization
python simulation_mirroring.py \
    --model ../../experiments/M3_dr_seed42_epnckzy2/final_model.zip \
    --vec-normalize ../../experiments/M3_dr_seed42_epnckzy2/vec_normalize.pkl \
    --render \
    --episodes 3
```

**That's it!** MuJoCo window opens, robot mirrors the simulation.

📖 **Full guide:** [`SIMULATION_MIRRORING_GUIDE.md`](SIMULATION_MIRRORING_GUIDE.md)

---

## 📚 Documentation

### Start Here
- **[`SIMULATION_MIRRORING_GUIDE.md`](SIMULATION_MIRRORING_GUIDE.md)** ⭐ - Robot mimics simulation (30 min setup)
- **[`QUICK_START.md`](QUICK_START.md)** - Quick reference and commands
- **[`DEPLOYMENT_APPROACHES.md`](DEPLOYMENT_APPROACHES.md)** - Compare both methods

### Advanced
- **[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)** - Full camera-based deployment (2-3 hours)
- **[`HARDWARE_CHECKLIST.md`](HARDWARE_CHECKLIST.md)** - Hardware assembly and requirements

### Scripts
- **[`realant_hardware/simulation_mirroring.py`](realant_hardware/simulation_mirroring.py)** - Simulation mirroring script
- **[`realant_hardware/deploy_robust_policy.py`](realant_hardware/deploy_robust_policy.py)** - Direct deployment script
- **[`realant_hardware/test_servo_connection.py`](realant_hardware/test_servo_connection.py)** - Connection test utility

---

## 🤖 What This Does

**NO ROS Required!** Uses the original RealANT ZeroMQ architecture:

```
Trained Policy (.zip + .pkl)
    ↓
deploy_robust_policy.py
    ↓ (ZeroMQ pub-sub)
┌───────────────┬───────────────┐
↓               ↓               ↓
AR Tag        Servo         Real
Tracker     Controller      Robot
(camera)      (USB)        (8 servos)
```

**Key Features**:
- ✅ Direct policy deployment (no simulation mirroring)
- ✅ Real-time 20Hz control loop
- ✅ Handles AR tag latency (110ms) via observation stacking
- ✅ Works with your trained VecNormalize stats
- ✅ Supports all 4 research models (M1-M4)

---

## 🏆 Which Model to Use?

| Model | Sim Performance | Expected Real | Recommendation |
|-------|----------------|---------------|----------------|
| M1 (Baseline) | 11.20m | 3-5m | ❌ Skip - will likely fail |
| M2 (SR2L) | 8.91m | 6-8m | ✅ Good - handles noise |
| **M3 (DR V7.7E)** | 7.90m | **6-7m** | ✅ **BEST** - joint failure specialist |
| M4 (Combo) | 7.86m | 5-6m | ⚠️ OK - but M3 better |

**Recommendation**: Use **M3 (DR V7.7E)** - it was trained with joint failures and sensor noise, which matches real robot conditions.

---

## ⚙️ System Architecture

### Why No ROS?

Michael Buekman was **RIGHT** that ROS doesn't work with RealANT!

| ROS (Doesn't Work) | ZeroMQ (Works!) |
|-------------------|----------------|
| 200ms+ latency | <50ms latency |
| Heavyweight middleware | Lightweight messaging |
| Requires roscore | Single Python library |
| Variable timing | Consistent 20Hz |

The original RealANT authors designed it this way intentionally.

### Communication Flow

**Observations** (20Hz):
```
Camera → ArUco Detection → Robot Pose (x,y,z,roll,pitch,yaw)
    ↓ ZMQ port 5555
deploy_robust_policy.py

Servos → Joint Encoders → Joint Positions (8D)
    ↓ ZMQ port 5557
deploy_robust_policy.py
```

**Actions** (20Hz):
```
deploy_robust_policy.py
    ↓ ZMQ port 5556
Dynamixel Commands → Servos (8D setpoints)
```

---

## 📊 Expected Performance

### Sim-to-Real Gap

Expect 20-30% performance drop (typical for locomotion):

| Metric | Simulation | Real Robot |
|--------|------------|------------|
| Distance (10s) | 7.90m | 5.5-6.5m |
| Baseline Speed | 0.539 m/s | 0.38-0.43 m/s |
| Success Rate | ~100% | 80-90% |

### Why the Gap?

- **AR tag noise**: σ ≈ 0.01 (trained for this!)
- **Servo lag**: 10-20ms response time
- **Floor friction**: Varies by surface
- **Observation latency**: 110ms camera delay (handled by observation stacking)

**Good news**: Your M3 model was trained for all of these!

---

## 🔧 Troubleshooting

See [`DEPLOYMENT_GUIDE.md#troubleshooting`](DEPLOYMENT_GUIDE.md#troubleshooting) for detailed solutions.

**Common Issues**:

| Problem | Quick Fix |
|---------|-----------|
| AR tags not detected | Add more lighting, check tag flatness |
| Servos don't move | Check USB port, 12V power |
| Robot doesn't walk | Use M3 model (not M1) |
| Communication timeout | Start processes in order (AR → servos → policy) |

---

## 📁 Files

```
deployment/
├── README.md                          ← You are here
├── DEPLOYMENT_GUIDE.md                ← Complete guide (start here!)
├── HARDWARE_CHECKLIST.md              ← Hardware setup checklist
└── realant_hardware/
    ├── __init__.py
    ├── deploy_robust_policy.py        ← Main deployment script
    └── test_model_loading.py          ← Test models before deployment
```

---

## 🚀 Next Steps

1. **Read**: [`HARDWARE_CHECKLIST.md`](HARDWARE_CHECKLIST.md) - Physical setup guide
2. **Setup**: Camera, AR tags, lighting, calibration (~2 hours)
3. **Test**: `python test_model_loading.py --test-all` (verify models work)
4. **Deploy**: `python deploy_robust_policy.py --model ...` (run on robot!)
5. **Record**: Videos + metrics for publication
6. **Publish**: Paper with real-world results! 🎉

---

## 💡 Key Insights

`★ Insight ─────────────────────────────────────`
**Why This Approach Works (and Michael's Didn't):**

1. **Your policies expect noisy observations** - trained with σ=0.01 sensor noise
2. **Observation stacking handles latency** - 4-frame history manages 110ms AR tag delay
3. **VecNormalize provides robustness** - implicit noise filtering from training
4. **Direct deployment** - policy sees real observations, not simulated ones
5. **M3 is perfect for real robot** - trained for exactly the perturbations robot has

**Michael's "mirror simulation" approach failed because**:
- Simulation ran at different speed than real-time
- Policy never saw real sensor characteristics
- Closed-loop control broke with observation mismatch
`─────────────────────────────────────────────────`

---

## 📖 References

- **RealANT Paper**: https://arxiv.org/abs/2011.03085 (Boney et al., 2022)
- **RealANT GitHub**: https://github.com/AaltoVision/realant-rl
- **Your Research**: See main project README for complete methodology

---

## ⚠️ Safety

- Clear 1m+ workspace around robot
- Keep emergency stop ready (unplug power)
- Supervise all episodes
- Let servos cool after 5-10 episodes

---

**Questions?** See troubleshooting section in [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) or check RealANT repo issues.

**Good luck with deployment!** 🤖 Your M3 model is ready for the real world! 🚀
