# RealANT Hardware Setup Checklist

Quick reference for setting up the physical RealANT robot for policy deployment.

---

## 🛒 Shopping List (What You Need to Buy)

### Camera Setup
- [ ] **Logitech Brio 4K Webcam** (~$200) OR similar 1080p 60fps camera
- [ ] **Tripod** (1.5m+ height) OR overhead mounting rig
- [ ] **USB extension cable** (3-5m) for camera

### Lighting
- [ ] **2-3× LED Floodlights** (20-50W each, ~$20-40 each)
  - ⚠️ Must be LED (NOT fluorescent - causes flicker)
- [ ] **Power strip** for lights

### AR Tags & Calibration
- [ ] **Cardboard sheets** (for backing AR tags)
- [ ] **Printer** with black/white printing
- [ ] **Chessboard pattern** (print from OpenCV docs)
- [ ] **Double-sided tape** or mounting adhesive
- [ ] **Clear tape** (for floor reference tag)

### Robot (Already Have)
- [x] RealANT robot from Ote Robotics
- [x] 8× Dynamixel AX-12A servos
- [x] OpenCM9.04-A board (or Arduino Nano 33 IoT)
- [x] USB cable (robot to computer)
- [x] 12V 5A power supply

### Computer
- [ ] Laptop/desktop with:
  - USB ports (2+)
  - Python 3.8+
  - 8GB+ RAM
  - Linux, macOS, or Windows

---

## 📸 Camera Setup (30 min)

### Step 1: Mount Camera
```
Target Configuration:
        📷 Camera
        │
        │ 1.0-1.5m
        │
        ▼
    ┌──────────┐
    │  Robot   │  ← Looking straight down
    │ Workspace│
    └──────────┘
```

- [ ] Mount camera on tripod at ~1.2m height
- [ ] Point camera **straight down** (bird's eye view)
- [ ] Ensure stable (robot vibrations won't shake it)
- [ ] Connect USB cable to computer
- [ ] Test camera: `ls /dev/video*` (Linux) or check Camera app

**Coverage Test**: Can you see a 1m × 1m area below camera?

---

### Step 2: Position Lighting
```
Setup:
    💡          💡
     ↘        ↙
      ┌──────┐
      │Robot │
      │Space │
      └──────┘
```

- [ ] Place 2-3 LED floodlights around workspace
- [ ] Angle to eliminate shadows on robot
- [ ] **Important**: NO fluorescent lights (cause flicker)
- [ ] Bright enough to read printed text easily

**Test**: Take photo with camera - can you clearly see robot details?

---

## 🏷️ AR Tag Preparation (15 min)

### Step 1: Generate Tags
1. [ ] Visit: https://chev.me/arucogen/
2. [ ] Generate **Marker ID 0** (for robot)
3. [ ] Generate **Marker ID 1** (for floor reference)
4. [ ] Settings:
   - Dictionary: `DICT_4X4_50`
   - Size: 100mm (10cm)
   - Print at **100% scale** (NO SCALING!)

### Step 2: Prepare Tags
- [ ] Print both markers on white paper
- [ ] Glue to cardboard backing (for rigidity)
- [ ] Cut out with 1cm border
- [ ] Verify size: 10cm × 10cm (measure!)

### Step 3: Attach Tags

**Robot Tag (ID 0)**:
- [ ] Attach to **top center** of robot body
- [ ] Use double-sided foam tape (allows removal)
- [ ] Ensure tag is **flat** and **horizontal**
- [ ] **Important**: Black/white pattern must face UP (toward camera)

**Reference Tag (ID 1)**:
- [ ] Place on floor in robot workspace
- [ ] Tape securely with clear tape
- [ ] Should be **always visible** to camera
- [ ] Robot should NOT walk on top of this tag

---

## 🔧 Robot Connection (10 min)

### Physical Connections
```
Servo Chain: Servo1 ↔ Servo2 ↔ ... ↔ Servo8
                ↓ (3-pin cables)
        OpenCM9.04 Board
                ↓ (USB)
            Computer
        (12V power supply)
```

- [ ] Connect all 8 servos in daisy-chain
- [ ] Plug servos into OpenCM board (4 leg ports)
- [ ] Connect **12V power supply** to board
- [ ] Connect **USB cable** from board to computer
- [ ] Check: LED on board should light up

### Test Connection
```bash
# Linux: Find device
ls /dev/ttyUSB* /dev/ttyACM*

# macOS: Find device
ls /dev/tty.usbserial-*

# Should see something like:
/dev/ttyUSB0  ← Use this port!
```

- [ ] Note device path (you'll need it later)
- [ ] Servos should have power (may move slightly on startup)

---

## 🎯 Camera Calibration (20 min)

### Step 1: Print Chessboard
- [ ] Download pattern: https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png
- [ ] Print at 100% scale (4cm squares)
- [ ] Glue to flat cardboard

### Step 2: Run Calibration
```bash
cd ~/realant-rl
python calibrate_camera.py --camera 0 --output cam_calib.pkl
```

- [ ] Hold chessboard in different positions under camera
- [ ] Tilt at various angles (flat, 30°, 45°)
- [ ] Move to different positions in workspace
- [ ] Collect 20-30 images (press space to capture)
- [ ] Press 'q' when done
- [ ] Check: `cam_calib.pkl` file created

---

## ✅ Pre-Deployment Verification

### Test 1: AR Tag Tracking
```bash
cd ~/realant-rl
python showaruco_board.py --camera 0 --calibration cam_calib.pkl
```

**Check**:
- [ ] Camera view appears
- [ ] Both tags (ID 0 and ID 1) detected
- [ ] Position (x, y, z) displayed smoothly
- [ ] Orientation (roll, pitch, yaw) displayed
- [ ] Move robot tag - tracking follows smoothly
- [ ] No flickering or tag loss

**If tags not detected**:
- ❌ Improve lighting (add more LEDs)
- ❌ Check tag quality (print quality, flatness)
- ❌ Adjust camera exposure (lower = less motion blur)
- ❌ Recalibrate camera

---

### Test 2: Servo Control
```bash
cd ~/realant-rl
python ant_server.py --port /dev/ttyUSB0 --test
```

**Check**:
- [ ] All 8 servos detected (IDs 1-8)
- [ ] Can read joint positions
- [ ] Servos respond to test commands (move slightly)
- [ ] No errors or timeouts

**If servos don't work**:
- ❌ Check USB device path (try /dev/ttyACM0)
- ❌ Check 12V power connected
- ❌ Check servo daisy-chain cables
- ❌ Use Dynamixel Wizard to verify servo IDs

---

### Test 3: Model Loading
```bash
source /path/to/your/venv/bin/activate
cd ~/realant-rl

python test_model_loading.py \
    --model /path/to/done/v7_7e_ultra_speed_jtfwl2qf/best_model/best_model.zip \
    --vec-normalize /path/to/done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl
```

**Check**:
- [ ] Model loads without errors
- [ ] VecNormalize loads without errors
- [ ] Observation space is 29D
- [ ] Action space is 8D
- [ ] Inference produces valid actions

---

## 🚀 Ready for Deployment!

If ALL checks pass above:
- ✅ Camera tracking working
- ✅ Servos responding
- ✅ Model loading correctly

You're ready to deploy! Proceed to:
```bash
python deploy_robust_policy.py \
    --model /path/to/model.zip \
    --vec-normalize /path/to/vec_normalize.pkl
```

See `DEPLOYMENT_GUIDE.md` for full deployment instructions.

---

## 📊 Workspace Layout Diagram

```
Top View:
┌─────────────────────────────────────┐
│          Overhead Camera 📷         │
│               ↓ 1.2m                │
│                                     │
│  💡                         💡      │
│    ┌───────────────────┐            │
│    │                   │            │
│    │  [1]  Reference   │            │
│    │  tag (floor)      │            │
│    │                   │            │
│    │       🤖          │            │
│    │     Robot         │            │
│    │     [0] tag       │            │
│    │                   │            │
│    │   Workspace       │            │
│    │   (1m × 1m)       │            │
│    └───────────────────┘            │
│                                     │
│  💡 Lights around edges      💡     │
│                                     │
│  Computer: 💻                       │
│  - USB to robot                     │
│  - USB to camera                    │
└─────────────────────────────────────┘
```

---

## ⚠️ Safety Notes

- [ ] **Clear workspace** of obstacles (1m+ clearance)
- [ ] **Emergency stop**: Know where power plug is (unplug if needed)
- [ ] **Servo overheating**: Let robot cool after 5-10 episodes
- [ ] **Cable management**: Tape cables to floor (trip hazard)
- [ ] **Supervision**: Never leave robot running unattended

---

## 🔧 Troubleshooting Quick Reference

| Problem | Quick Fix |
|---------|-----------|
| Tags not detected | Add more lighting, lower camera exposure |
| Tags flicker | Check tag flatness, improve lighting |
| Servos don't respond | Check USB port, 12V power, cables |
| Robot doesn't walk | Use M3 model (not M1), check starting position |
| Communication timeout | Start AR tracker first, then servos, then deployment |
| Robot walks backward | Rotate reference tag 180° |

---

## 📝 Shopping Links (Examples)

**Camera**:
- Logitech Brio 4K: https://www.logitech.com/en-us/products/webcams/brio-4k-hdr-webcam.960-001105.html
- Alternative: Logitech C920 (1080p 30fps) - cheaper

**Lighting**:
- LED Work Light (example): Search "LED work light 50W" on Amazon
- Need 2-3 lights, ~$25 each

**Tripod**:
- Any photo/video tripod that extends to 1.5m+
- ~$30-50 on Amazon

**Printables**:
- AR Tags: https://chev.me/arucogen/
- Chessboard: https://docs.opencv.org/4.x/pattern.png

---

**Time Budget**:
- Hardware setup: 30 min
- AR tag prep: 15 min
- Calibration: 20 min
- Testing: 30 min
- **Total: ~2 hours to full deployment**

Good luck! 🚀
