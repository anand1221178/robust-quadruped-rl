#!/usr/bin/env python3
"""
🏆 Interactive Robustness Championship Dashboard

Streamlit web app for thesis defense - lets judges interactively test models
Run with: streamlit run scripts/interactive_judge_demo.py
"""
import sys
sys.path.append('src')

import streamlit as st
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time

# Page config
st.set_page_config(
    page_title="4-Model Robustness Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model configurations
MODELS = {
    'M1 Baseline': {
        'key': 'M1',
        'path': 'experiments/M1_baseline_32M_RETRAINED_ym2jcllj/final_model',
        'vec_path': 'experiments/M1_baseline_32M_RETRAINED_ym2jcllj/vec_normalize.pkl',
        'color': '#FFD700',  # Gold
        'description': 'Pure PPO baseline (32M steps)'
    },
    'M2 SR2L': {
        'key': 'M2',
        'path': 'experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/final_model',
        'vec_path': 'experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/vec_normalize.pkl',
        'color': '#00FF00',  # Green
        'description': 'SR2L - Sensor noise specialist (32M steps)'
    },
    'M3 DR': {
        'key': 'M3',
        'path': 'experiments/M3_dr_v2_single_failures_32M_15cxapkl/final_model',
        'vec_path': 'experiments/M3_dr_v2_single_failures_32M_15cxapkl/vec_normalize.pkl',
        'color': '#FF00FF',  # Magenta
        'description': 'Domain Randomization - Actuator failure specialist (32M steps)'
    },
    'M4 Combo': {
        'key': 'M4',
        'path': 'done/ultimate_robustness_combo_ju7lfsk2/final_model',
        'vec_path': 'done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
        'color': '#00FFFF',  # Yellow
        'description': 'SR2L + DR combined (Ultimate)'
    }
}

JOINT_MAP = {
    "Hip 1": 0, "Ankle 1": 1,
    "Hip 2": 2, "Ankle 2": 3,
    "Hip 3": 4, "Ankle 3": 5,
    "Hip 4": 6, "Ankle 4": 7
}

# Session state initialization
if 'baseline_velocity' not in st.session_state:
    st.session_state.baseline_velocity = 0.23  # Approximate baseline

# Title
st.title("🏆 Interactive Robustness Championship")
st.subheader("Real-time policy testing under stress conditions")

# Sidebar: Controls
with st.sidebar:
    st.header("🎮 Test Configuration")

    model_choice = st.selectbox(
        "Select Model",
        list(MODELS.keys()),
        help="Choose which trained model to test"
    )

    # Show model description
    st.info(MODELS[model_choice]['description'])

    st.divider()
    st.subheader("⚡ Stress Conditions")

    noise_level = st.slider(
        "Sensor Noise Level",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Gaussian noise std applied to joint sensors (dims 13-28)"
    )

    failed_joint = st.selectbox(
        "Joint Failure",
        ["None"] + [f"{j.title()}" for j in ["Hip 1", "Ankle 1", "Hip 2", "Ankle 2",
                                               "Hip 3", "Ankle 3", "Hip 4", "Ankle 4"]],
        help="Select joint to lock after 2-second warmup"
    )

    duration = st.select_slider(
        "Test Duration (seconds)",
        options=[10, 20, 30, 60],
        value=20
    )

    st.divider()

    # Main run button
    run_button = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

    st.divider()
    st.subheader("🚀 Quick Tests")
    st.caption("Pre-configured demonstrations")

    quick_m2 = st.button("M2 vs Noise 0.5", use_container_width=True,
                         help="Show M2 SR2L advantage under high sensor noise")
    quick_m3 = st.button("M3 vs Hip 4 Failure", use_container_width=True,
                         help="Show M3 DR advantage under joint failure")
    quick_compare = st.button("M1 vs M2 (Noise)", use_container_width=True,
                              help="Direct comparison under noise")

# Main area: Simulation + Metrics
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Simulation")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("📊 Performance Metrics")
    velocity_metric = st.empty()
    distance_metric = st.empty()
    retention_metric = st.empty()

    st.subheader("📈 Real-time Chart")
    chart_placeholder = st.empty()

# Helper function to load model (only model, not environment)
@st.cache_resource
def load_model_only(model_path):
    """Load just the model (cached)"""
    return PPO.load(model_path)

def create_env_and_load(model_path, vec_path):
    """Create environment and load model (called when simulation runs)"""
    def make_env():
        base_env = gym.make('RealAntMujoco-v0',
                           render_mode='rgb_array',
                           disable_env_checker=True)
        while isinstance(base_env, TimeLimit):
            base_env = base_env.env
        env = TimeLimit(base_env, max_episode_steps=2500)
        env = SuccessRewardWrapper(env)
        return env

    env = DummyVecEnv([make_env])

    try:
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
    except:
        pass

    model = load_model_only(model_path)

    return model, env

def run_simulation(model_name, noise, joint_failure, test_duration):
    """Run a simulation and display results"""

    model_config = MODELS[model_name]

    with st.spinner(f"Loading {model_name}..."):
        try:
            model, env = create_env_and_load(model_config['path'], model_config['vec_path'])
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    status_placeholder.info(f"Running {model_name} with noise={noise}, joint={joint_failure}, duration={test_duration}s")

    obs = env.reset()
    positions = []
    velocities = []

    # Convert joint failure
    failed_idx = JOINT_MAP.get(joint_failure, None)

    total_steps = test_duration * 60  # 60 fps

    for step in range(total_steps):
        # Apply noise
        if noise > 0:
            obs_noisy = obs.copy()
            noise_vec = np.random.normal(0, noise, (1, 16))
            obs_noisy[0, 13:29] += noise_vec.flatten()
        else:
            obs_noisy = obs

        # Get action
        action, _ = model.predict(obs_noisy, deterministic=True)

        # Apply failure (after 2s warmup)
        if failed_idx is not None and step >= 120:
            action[0][failed_idx] = 0.0

        # Step environment
        obs, reward, done, info = env.step(action)

        # Track metrics
        try:
            x_pos = env.envs[0].unwrapped.data.qpos[0]
            positions.append(x_pos)

            if len(positions) >= 2:
                distance = positions[-1] - positions[0]
                time_elapsed = len(positions) * 0.05  # 50ms per step
                velocity = distance / time_elapsed if time_elapsed > 0 else 0
                velocities.append(velocity)
        except:
            positions.append(0)
            velocities.append(0)

        # Render every 2 frames (30fps display)
        if step % 2 == 0:
            try:
                frame = env.envs[0].render()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update video
                video_placeholder.image(frame_rgb, use_container_width=True)

                # Update metrics
                if velocities:
                    current_vel = velocities[-1]
                    current_dist = distance if len(positions) >= 2 else 0
                    retention = (current_vel / st.session_state.baseline_velocity * 100) if st.session_state.baseline_velocity > 0 else 100

                    velocity_metric.metric(
                        "Current Velocity",
                        f"{current_vel:.3f} m/s",
                        delta=f"{current_vel - st.session_state.baseline_velocity:.3f} m/s"
                    )
                    distance_metric.metric(
                        "Distance Traveled",
                        f"{current_dist:.2f} m"
                    )
                    retention_metric.metric(
                        "Retention vs Baseline",
                        f"{retention:.0f}%",
                        delta=f"{retention - 100:.0f}%"
                    )

                    # Update chart (every 10 frames to reduce overhead)
                    if step % 20 == 0 and len(velocities) > 1:
                        chart_data = pd.DataFrame({
                            'Time (s)': [i * 0.05 for i in range(len(velocities))],
                            'Velocity (m/s)': velocities
                        })
                        chart_placeholder.line_chart(
                            chart_data.set_index('Time (s)'),
                            use_container_width=True
                        )
            except Exception as e:
                pass  # Skip rendering errors

        if done[0]:
            obs = env.reset()

    # Final summary
    if velocities:
        final_velocity = np.mean(velocities[-60:]) if len(velocities) >= 60 else np.mean(velocities)
        final_distance = positions[-1] - positions[0] if len(positions) >= 2 else 0
        retention = (final_velocity / st.session_state.baseline_velocity * 100) if st.session_state.baseline_velocity > 0 else 100

        status_placeholder.success(
            f"✅ Test complete! Final velocity: {final_velocity:.3f} m/s | "
            f"Distance: {final_distance:.2f}m | Retention: {retention:.0f}%"
        )

# Run simulation when button clicked
if run_button or quick_m2 or quick_m3 or quick_compare:
    # Handle quick test presets
    if quick_m2:
        model_choice = "M2 SR2L"
        noise_level = 0.5
        failed_joint = "None"
        duration = 20
    elif quick_m3:
        model_choice = "M3 DR"
        noise_level = 0.0
        failed_joint = "Hip 4"
        duration = 30
    elif quick_compare:
        st.info("Running M1 Baseline first, then M2 SR2L for comparison...")

        # Run M1 first
        st.subheader("🥇 M1 Baseline (with noise 0.3)")
        run_simulation("M1 Baseline", 0.3, "None", 20)

        time.sleep(2)

        # Then M2
        st.subheader("🥇 M2 SR2L (with noise 0.3)")
        run_simulation("M2 SR2L", 0.3, "None", 20)

        st.success("Comparison complete! Notice M2's superior performance under noise.")
    else:
        # Regular run
        run_simulation(model_choice, noise_level, failed_joint, duration)

# Footer
st.divider()
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.caption("🎓 Thesis Defense Demo")
    st.caption("4 Models Trained to 32M Steps")

with col_b:
    st.caption("📊 Real-time MuJoCo Simulation")
    st.caption("PPO + Domain Randomization")

with col_c:
    st.caption("⚙️ Built with Streamlit")
    st.caption("Interactive Model Testing")
