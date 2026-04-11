# Quick Start: RealANT Deployment

**Get your trained models running on the physical robot in under 30 minutes!**

---

## Prerequisites

- [ ] RealANT robot assembled
- [ ] 8× Dynamixel servos connected
- [ ] OpenCM board or Arduino Nano 33 IoT flashed with firmware
- [ ] USB cable connecting board to laptop
- [ ] 12V power supply connected

---

## Step 1: Choose Your Approach (2 minutes)

### Option A: Simulation Mirroring (RECOMMENDED FOR FIRST TIME)
- ✅ Quick setup (30 min total)
- ✅ See MuJoCo visualization
- ✅ No camera/AR tags needed
- ⚠️ Open-loop control (may veer off course)

### Option B: Direct Policy Deployment
- ⚠️ Longer setup (2-3 hours first time)
- ✅ Best performance
- ✅ Closed-loop control
- ❌ Requires camera + AR tags + calibration

**For your first deployment, use Option A!**

---

## Step 2: Install Dependencies (5 minutes)

```bash
cd /Users/anand/Documents/4th\ Year/robust-quadruped-rl

# Activate your venv if you have one
source venv/bin/activate

# Install additional dependencies
pip install pyzmq opencv-python
```

---

## Step 3: Test Servo Connection (5 minutes)

Find your serial port:

```bash
# macOS
ls /dev/tty.usbserial-* /dev/tty.usbmodem* /dev/cu.usbserial-* /dev/cu.usbmodem*

# Linux
ls /dev/ttyUSB* /dev/ttyACM*
```

You should see something like:
- macOS: `/dev/tty.usbserial-0001` or `/dev/cu.usbmodem14101`
- Linux: `/dev/ttyUSB0` or `/dev/ttyACM0`

Test servo communication:

```bash
cd deployment/realant_hardware/realant

# Replace with YOUR serial port
python ant_server.py --port /dev/tty.usbserial-0001
```

**Expected output:**
```
Connecting to servos...
✅ Found 8 servos: [1, 2, 3, 4, 5, 6, 7, 8]
Listening for commands on port 5556...
```

If you see this, **GREAT!** Your servos are working. Leave this terminal running.

---

## Step 4: Run Simulation Mirroring (10 minutes)

Open a **NEW terminal window**:

```bash
cd /Users/anand/Documents/4th\ Year/robust-quadruped-rl/deployment/realant_hardware

# Run M3 (best model) with visualization
python simulation_mirroring.py \
    --model ../../experiments/M3_dr_seed42_epnckzy2/best_model.zip \
    --vec-normalize ../../experiments/M3_dr_seed42_epnckzy2/vec_normalize.pkl \
    --render \
    --episodes 1
```

### What Will Happen:

1. **Script loads** - You'll see initialization messages
2. **MuJoCo window opens** - Shows simulated RealANT
3. **Prompt appears** - "Press Enter to start episode..."
4. **Place robot** in starting position (lying down, legs splayed)
5. **Press Enter**
6. **Watch:**
   - MuJoCo window: Simulated robot walks forward
   - Physical robot: Mirrors the simulation!

### Troubleshooting

**"Model not found":**
```bash
# Check if model exists
ls ../../experiments/M3_dr_seed42_epnckzy2/

# If best_model.zip doesn't exist, use final_model.zip:
python simulation_mirroring.py \
    --model ../../experiments/M3_dr_seed42_epnckzy2/final_model.zip \
    --vec-normalize ../../experiments/M3_dr_seed42_epnckzy2/vec_normalize.pkl \
    --render
```

**"No module named 'realant_sim'":**
```bash
# Make sure you're in the project directory
cd /Users/anand/Documents/4th\ Year/robust-quadruped-rl
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

**"ZeroMQ connection failed":**
- Make sure `ant_server.py` is still running in Terminal 1
- Check that port 5556 is not blocked by firewall

**"Robot doesn't move":**
- Check power supply is connected and ON
- Verify USB cable is connected
- Try unplugging/replugging USB cable

---

## Step 5: Try Other Models (Optional)

Compare all 4 models:

### M1 - Baseline (No Robustness Training)
```bash
python simulation_mirroring.py \
    --model ../../experiments/M1_baseline_seed42_rap72nn2/final_model.zip \
    --vec-normalize ../../experiments/M1_baseline_seed42_rap72nn2/vec_normalize.pkl \
    --render \
    --episodes 1
```

Expected: May walk in simulation, but real robot likely struggles (no robustness)

### M2 - SR2L (Sensor Noise Robustness)
```bash
python simulation_mirroring.py \
    --model ../../experiments/M2_sr2l_seed42_8qknaqel/final_model.zip \
    --vec-normalize ../../experiments/M2_sr2l_seed42_8qknaqel/vec_normalize.pkl \
    --render \
    --episodes 1
```

Expected: Moderate performance (trained for sensor noise, not dynamics mismatch)

### M3 - Domain Randomization (BEST MODEL)
```bash
python simulation_mirroring.py \
    --model ../../experiments/M3_dr_seed42_epnckzy2/best_model.zip \
    --vec-normalize ../../experiments/M3_dr_seed42_epnckzy2/vec_normalize.pkl \
    --render \
    --episodes 1
```

Expected: **Best performance** (trained for joint failures, most robust)

### M4 - Combined SR2L+DR
```bash
python simulation_mirroring.py \
    --model ../../experiments/M4_combo_seed42_p0n7r1gw/final_model.zip \
    --vec-normalize ../../experiments/M4_combo_seed42_p0n7r1gw/vec_normalize.pkl \
    --render \
    --episodes 1
```

Expected: Good, but likely worse than M3 alone (negative interference from combo)

---

## Step 6: What's Next?

### If Simulation Mirroring Works Well:
✅ Your hardware is working!
✅ Your models are loaded correctly!

**Next steps:**
1. Try all 4 models and compare
2. Record videos (side-by-side sim + robot)
3. Move to **Direct Policy Deployment** for best results

### If Robot Performance is Poor:
This is expected for simulation mirroring! The robot is running open-loop (no sensor feedback).

**To improve performance:**
- Set up camera + AR tags (see `DEPLOYMENT_GUIDE.md`)
- Switch to direct policy deployment (`deploy_robust_policy.py`)
- This gives the policy REAL sensor feedback to adapt

---

## Model Paths Reference

All your trained models are here:
```
/Users/anand/Documents/4th Year/robust-quadruped-rl/experiments/
```

**M1 - Baseline:**
- `M1_baseline_seed42_rap72nn2/`
- `M1_baseline_seed123_ef86brh7/`
- `M1_baseline_seed456_s39mzd6e/`
- `M1_baseline_seed789_b6pielh9/`
- `M1_baseline_seed999_uzo3vstk/`

**M2 - SR2L:**
- `M2_sr2l_seed42_8qknaqel/`
- `M2_sr2l_seed123_wgn4flak/`
- `M2_sr2l_seed456_qjwh15rq/`
- `M2_sr2l_seed789_g9loing3/`
- `M2_sr2l_seed999_p977kpg4/`

**M3 - Domain Randomization (BEST):**
- `M3_dr_seed42_epnckzy2/` ⭐ **START WITH THIS ONE**
- `M3_dr_seed123_yxt6lw5f/`
- `M3_dr_seed456_ijv5yz5g/`
- `M3_dr_seed789_howl2uf6/`
- `M3_dr_seed999_vzzp16pn/`

**M4 - Combined:**
- `M4_combo_seed42_p0n7r1gw/`
- `M4_combo_seed123_06fqvwf0/`
- `M4_combo_seed456_osu6bibl/`
- `M4_combo_seed789_pfcyh0d8/`
- `M4_combo_seed999_71xtry0k/`

Each directory contains:
- `best_model.zip` or `final_model.zip` - Trained policy weights
- `vec_normalize.pkl` - Observation normalization statistics (CRITICAL!)

---

## Common Issues

### "ImportError: No module named 'gymnasium'"
```bash
pip install gymnasium
```

### "ImportError: No module named 'mujoco'"
```bash
pip install mujoco
```

### "Cannot open MuJoCo renderer"
This happens on macOS sometimes. Try:
```bash
# Run without rendering (faster anyway)
python simulation_mirroring.py \
    --model ../../experiments/M3_dr_seed42_epnckzy2/best_model.zip \
    --vec-normalize ../../experiments/M3_dr_seed42_epnckzy2/vec_normalize.pkl \
    --no-render
```

### "Serial port permission denied" (Linux)
```bash
sudo chmod 666 /dev/ttyUSB0
# Or add yourself to dialout group:
sudo usermod -a -G dialout $USER
# Then logout and login again
```

---

## Success Checklist

After completing this quick start, you should have:
- [x] Verified servo communication
- [x] Loaded and run M3 model
- [x] Seen simulation visualization
- [x] Physical robot mirrored simulation
- [x] Basic understanding of sim-to-real gap

**Congratulations! You're ready for more advanced deployment.** 🎉

---

## Next Steps

1. **Record Videos**
   - Screen record MuJoCo window
   - Record physical robot with phone/camera
   - Compare side-by-side

2. **Try All Models**
   - Run M1, M2, M3, M4
   - Note which works best
   - Should confirm M3 > M2 > M4 > M1

3. **Move to Direct Deployment**
   - Follow `DEPLOYMENT_GUIDE.md` for full setup
   - Set up camera + AR tags
   - Get closed-loop control for best results

4. **Collect Data for Paper**
   - 5 episodes per model
   - Measure distance traveled
   - Compare to simulation results
   - Quantify sim-to-real gap

---

## Quick Reference Commands

**Terminal 1 (always running):**
```bash
cd deployment/realant_hardware/realant
python ant_server.py --port /dev/tty.usbserial-0001
```

**Terminal 2 (run models):**
```bash
cd deployment/realant_hardware

# M3 model (best)
python simulation_mirroring.py \
    --model ../../experiments/M3_dr_seed42_epnckzy2/best_model.zip \
    --vec-normalize ../../experiments/M3_dr_seed42_epnckzy2/vec_normalize.pkl \
    --render \
    --episodes 3
```

**Stop everything:**
- Ctrl+C in both terminals
- Unplug robot power (servos will release)

---

**Need help?** Check:
- `DEPLOYMENT_APPROACHES.md` - Detailed comparison of deployment methods
- `DEPLOYMENT_GUIDE.md` - Full setup for direct policy deployment
- `HARDWARE_CHECKLIST.md` - Hardware requirements and assembly

**Good luck with your deployment!** 🚀🤖
