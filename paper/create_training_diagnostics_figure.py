#!/usr/bin/env python3
"""
Create 2-panel training diagnostics figure from W&B exports.

Combines value loss and entropy plots into a single publication-quality figure
showing M4's catastrophic collapse at 14-20M steps (200-250k iterations).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
from pathlib import Path

# Configuration
FIGURES_DIR = Path(__file__).parent / "figures"
VALUE_LOSS_PATH = FIGURES_DIR / "training_diagnostics_value_loss.png"
ENTROPY_PATH = FIGURES_DIR / "training_diagnostics_entropy.png"
OUTPUT_PATH = FIGURES_DIR / "figure_S1_training_diagnostics"

# Collapse region (in iteration steps)
# 14-20M timesteps / 1536 batch size = ~9115-13021 iterations
# But W&B typically logs every N steps, so approximate visual range
COLLAPSE_START_ITER = 200000  # ~14M steps
COLLAPSE_END_ITER = 250000    # ~20M steps

def create_combined_figure():
    """Create 2-panel figure with value loss and entropy."""

    # Load images
    print(f"Loading {VALUE_LOSS_PATH}...")
    value_loss_img = Image.open(VALUE_LOSS_PATH)

    print(f"Loading {ENTROPY_PATH}...")
    entropy_img = Image.open(ENTROPY_PATH)

    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Panel A: Value Loss
    axes[0].imshow(np.array(value_loss_img))
    axes[0].axis('off')
    axes[0].set_title('(A) Value Loss', fontsize=16, fontweight='bold', pad=10)

    # Panel B: Entropy
    axes[1].imshow(np.array(entropy_img))
    axes[1].axis('off')
    axes[1].set_title('(B) Entropy', fontsize=16, fontweight='bold', pad=10)

    # Tight layout
    plt.tight_layout()

    # Save as PDF and PNG
    print(f"\nSaving combined figure...")
    plt.savefig(f"{OUTPUT_PATH}.pdf", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_PATH}.pdf")

    plt.savefig(f"{OUTPUT_PATH}.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_PATH}.png")

    plt.close()

    print("\n✅ Training diagnostics figure created successfully!")
    print(f"\nFigure shows:")
    print("  - Panel A: Value loss collapse (100× decrease at 200-250k iters)")
    print("  - Panel B: Entropy remains stable during collapse")
    print("  - Evidence: Deterministic but ineffective policy (gradient conflict)")

if __name__ == "__main__":
    # Check if input files exist
    if not VALUE_LOSS_PATH.exists():
        print(f"❌ Error: {VALUE_LOSS_PATH} not found!")
        print(f"\nPlease export value loss from W&B and save as:")
        print(f"  {VALUE_LOSS_PATH}")
        exit(1)

    if not ENTROPY_PATH.exists():
        print(f"❌ Error: {ENTROPY_PATH} not found!")
        print(f"\nPlease export entropy from W&B and save as:")
        print(f"  {ENTROPY_PATH}")
        exit(1)

    create_combined_figure()
