#!/usr/bin/env python3
"""
🏆 Interactive Robustness Championship Dashboard (Simple Version)
Metrics-only demo without video rendering (avoids GLFW crashes)
Run with: python3 -m streamlit run scripts/interactive_judge_demo_simple.py
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
        'color': '#00FFFF',  # Cyan
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
    st.session_state.baseline_velocity = 0.18  # Baseline from championship results

# Title
st.title("🏆 Interactive Robustness Championship")
st.subheader("Real-time policy testing under stress conditions (Metrics View)")

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
    quick_m4 = st.button("M4 vs Noise 0.3", use_container_width=True,
                         help="Show M4 combo advantage")

# Main area: Metrics
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Performance Metrics")
    velocity_metric = st.empty()
    distance_metric = st.empty()
    retention_metric = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("📈 Real-time Chart")
    chart_placeholder = st.empty()

# Helper function to load model
@st.cache_resource
def load_model_only(model_path):
    """Load just the model (cached)"""
    return PPO.load(model_path)

def create_env_no_render(vec_path):
    """Create environment WITHOUT rendering (faster, no GLFW issues)"""
    def make_env():
        # NO render_mode - this avoids GLFW initialization
        base_env = gym.make('RealAntMujoco-v0',
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

    return env

def run_simulation(model_name, noise, joint_failure, test_duration):
    """Run a simulation and display results (metrics only, no video)"""

    model_config = MODELS[model_name]

    with st.spinner(f"Loading {model_name}..."):
        try:
            model = load_model_only(model_config['path'])
            env = create_env_no_render(model_config['vec_path'])
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
    progress_bar = st.progress(0)

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
                time_elapsed = len(positions) / 60.0  # 60 fps
                velocity = distance / time_elapsed if time_elapsed > 0 else 0
                velocities.append(velocity)
        except:
            positions.append(0)
            velocities.append(0)

        # Update metrics every 30 frames (twice per second)
        if step % 30 == 0:
            progress_bar.progress(step / total_steps)

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
                if step % 60 == 0 and len(velocities) > 1:
                    chart_data = pd.DataFrame({
                        'Time (s)': [i / 60.0 for i in range(len(velocities))],
                        'Velocity (m/s)': velocities
                    })
                    chart_placeholder.line_chart(
                        chart_data.set_index('Time (s)'),
                        use_container_width=True
                    )

        if done[0]:
            obs = env.reset()

    progress_bar.progress(1.0)

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
if run_button or quick_m2 or quick_m3 or quick_m4:
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
    elif quick_m4:
        model_choice = "M4 Combo"
        noise_level = 0.3
        failed_joint = "None"
        duration = 20

    # Run simulation
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
    st.caption("Metrics-Only Mode (No Video)")
