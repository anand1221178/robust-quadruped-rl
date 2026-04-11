# A Systematic Ablation Study of Sensor and Actuator Robustness Methods for Quadruped Locomotion

[![Paper](https://img.shields.io/badge/Paper-Research_Paper-blue)](anand_patel_2561034.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This repository contains the implementation and experimental results for our systematic ablation study comparing robustness methods for quadruped locomotion. We investigate whether combining **Smooth Regularized Reinforcement Learning (SR2L)** and **Domain Randomization (DR)**—targeting sensor noise and actuator failures respectively—produces synergy or interference when combined.

### Key Finding
**Combining robustness methods creates interference, not synergy.** Our combined approach (SR2L+DR) underperformed specialized domain randomization by 56-145% under stress conditions due to conflicting gradient signals during training.

## 📊 Main Results

Across **38,000 evaluation episodes**, we found:

| Model | Method | Baseline Performance | Joint Failure Robustness | Sensor Noise Robustness |
|-------|--------|---------------------|-------------------------|------------------------|
| **M1** | Baseline PPO | 6.40m | 26.7% retention | 83% retention |
| **M2** | SR2L Only | 7.26m | 30.7% retention | **100.7% retention** |
| **M3** | DR Only | **8.26m** | **53.3% retention** | 92.2% retention |
| **M4** | SR2L + DR | 6.70m | 50.6% retention | 91% retention |

**Surprising Discovery:** Domain randomization (M3) achieved the **best baseline performance** (8.26m) by acting as implicit regularization, contradicting assumptions that robustness training sacrifices nominal performance.

## 🤖 Experimental Setup

### Environment
- **Robot:** RealAnt quadruped (8 actuated joints - 4 legs × 2 joints each)
- **Simulator:** MuJoCo physics engine
- **Task:** Forward locomotion with velocity-based rewards
- **Observation Space:** 29-dimensional proprioceptive state
- **Action Space:** 8-dimensional joint position commands

### Training Configuration
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Architecture:** 2-layer MLP [128, 128] with tanh activation
- **Training Steps:** 32M timesteps per model
- **Seeds:** 5 random seeds (42, 123, 456, 789, 999) for statistical robustness
- **Compute:** ~847 GPU-hours total on NVIDIA RTX 4090

## 📁 Repository Structure

```
robust-quadruped-rl/
├── paper/
│   ├── main.tex                    # Research paper (LaTeX)
│   ├── figures/                    # All paper figures and visualizations
│   │   ├── realant_simulation_final.pdf
│   │   ├── joint_numbering_diagram.pdf
│   │   ├── figure_1_baseline_performance.pdf
│   │   ├── figure_2_noise_robustness.pdf
│   │   ├── figure_joint_failure_analysis.pdf
│   │   ├── figure_robustness_methods_comparison.pdf
│   │   ├── figure_5_joint_noise_ablation.pdf
│   │   └── learning_curves_reconstructed.pdf
│   └── LaTeX template files
│
├── Scripts/                        # Analysis and visualization scripts
│   ├── create_figure_*.py        # Figure generation scripts
│   ├── compute_statistics.py     # Statistical analysis
│   ├── analyze_m4_collapse.py    # M4 training failure analysis
│   └── render_ant_*.py           # Robot visualization
│
├── Videos/                        # Demo videos (21GB, local only)
│   └── [Evaluation videos - not included in GitHub]
│
└── README.md                      # This file
```

## 🚀 Key Contributions

1. **Negative Transfer in Multi-Objective Robustness**
   - Quantitative evidence that combining SR2L and DR creates interference
   - Combined approach underperforms specialized training by 56-145%
   - Conflicting gradient signals identified as root cause

2. **Observation Normalization as Critical Infrastructure**
   - Normalization alone provides 83% sensor noise retention
   - Without normalization: >99% performance collapse
   - SR2L adds only marginal improvement (+17.7%) over normalization

3. **Cross-Distribution Generalization**
   - SR2L policies maintain performance on unseen noise types
   - 106.6% retention on Poisson noise (never seen in training)
   - 112.8% retention on salt-and-pepper noise

## 📈 Experimental Results

### Experiment 1: Baseline Performance
- **M3 (DR)** achieves best baseline: 8.26m
- Domain randomization acts as regularization
- M4 shows catastrophic collapse at 14-20M steps

### Experiment 2: Sensor Noise Robustness
- All models maintain >90% performance at 10× training noise
- M2 (SR2L) shows stochastic resonance: 100.7% retention
- Observation normalization prerequisite for learning

### Experiment 3: Joint Failure Robustness
- **M3 dominates:** 53.3% hip retention, 41.7% ankle retention
- Ankle failures consistently harder than hip failures
- Ankle_4 identified as universal worst-case joint

### Experiment 4: Combined Stress Testing
- No synergy from combining methods
- M3 outperforms M4 by 56-145% under combined stress
- Specialized training superior to multi-objective

### Experiment 5: Joint-Noise Interaction
- Joint failures and sensor noise combine additively
- No synergistic interaction effects
- M3 maintains best worst-case robustness (13.7% at extreme stress)

## 🛠️ Reproducibility

### Statistical Methodology
- **Multi-seed training:** 5 seeds per model for robustness
- **Episode pairing:** Same random seeds across models for fair comparison
- **Statistical tests:** Paired t-tests with Bonferroni correction (α=0.05)
- **Total episodes:** 38,000 evaluation episodes across all experiments

### Training Details
All models trained with identical hyperparameters:
- Learning rate: 3×10⁻⁴
- Batch size: 1,536
- Epochs per update: 8
- Clip range: 0.15
- GAE λ: 0.97
- Discount γ: 0.995

## 💡 Key Insights

`★ Insight ─────────────────────────────────────`
1. **Domain randomization provides "free" robustness** - improves baseline performance through regularization
2. **Conflicting objectives create gradient warfare** - SR2L demands insensitivity while DR demands adaptation
3. **Specialization beats generalization** - train for hardest failure mode (joint failures) for best overall robustness
`─────────────────────────────────────────────────`

## 🎯 Deployment Recommendations

**Universal recommendation: Deploy M3 (Domain Randomization)**
- Best performance across ALL scenarios
- Highest baseline speed (8.26m)
- Excellent joint failure robustness (53.3% retention)
- Surprising cross-robustness to sensor noise (92.2% retention)

**When to consider SR2L:**
- Facing unknown/non-Gaussian noise distributions
- Need theoretical robustness guarantees
- Sensor characteristics fundamentally different from simulation

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{patel2025systematic,
  title={A Systematic Ablation Study of Sensor and Actuator Robustness Methods for Quadruped Locomotion},
  author={Patel, Anand},
  journal={University of the Witwatersrand Research Papers},
  year={2025},
  institution={University of the Witwatersrand}
}
```

## 🔬 Limitations & Future Work

### Current Limitations
- Simulation-only evaluation (no physical robot validation)
- Limited hyperparameter exploration (λ=0.001, σ=0.01 fixed)
- Hand-designed DR curriculum
- PPO-specific findings

### Future Directions
1. Physical robot validation of recommendations
2. Comprehensive hyperparameter sweeps
3. Alternative combination strategies (sequential training)
4. Theoretical analysis of method compatibility
5. Extension to other algorithms (SAC, TD3)

## 📧 Contact

**Author:** Anand Patel
**Email:** anand.patel@students.wits.ac.za
**Institution:** School of Computer Science and Applied Mathematics, University of the Witwatersrand

## 🙏 Acknowledgments

This research was conducted at the University of the Witwatersrand, Johannesburg. We thank the reviewers for their valuable feedback and the open-source community for the tools that made this work possible.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note:** The Videos folder (21GB) contains evaluation recordings and is maintained locally only. Contact the author for access to specific demonstration videos if needed.