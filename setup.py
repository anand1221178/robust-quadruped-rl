from setuptools import setup, find_packages

setup(
    name="robust-quadruped-rl",
    version="0.1.0",
    author="Anand Patel",
    description="Robust quadruped locomotion using RL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.28.0",
        "mujoco>=2.3.0",
    ],
)
