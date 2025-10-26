#!/usr/bin/env python3
"""
Curriculum Logging Callback
============================

Logs DR curriculum phase transitions and statistics to W&B.
Reads from info dict populated by CurriculumDRWrapper.

This callback bridges the gap between CurriculumDRWrapper (which tracks phases)
and W&B (which needs explicit logging). Without this, phase transitions happen
internally but don't show up in W&B logs.

Author: Auto-generated for final paper retraining runs
Date: October 26, 2025
"""

from stable_baselines3.common.callbacks import BaseCallback
import wandb


class CurriculumLoggingCallback(BaseCallback):
    """
    Logs curriculum phase transitions and DR statistics to W&B.

    Reads curriculum_phase, dropped_joints, and failure_probability from
    the info dict that CurriculumDRWrapper populates on each step.

    Creates visual markers in W&B at phase transitions for easy identification
    of when curriculum changes occurred during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_logged_phase = None
        self.phase_transition_steps = {}
        self.episode_count = 0

    def _on_training_start(self) -> None:
        """Log initial curriculum state"""
        if self.verbose > 0:
            print("✅ Curriculum logging callback initialized")
            print("   Will track phase transitions and log to W&B")

    def _on_step(self) -> bool:
        """Log curriculum metrics on every step"""

        # Get info from latest step
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]  # First environment's info dict

            # Extract curriculum information from info dict
            current_phase = info.get('curriculum_phase', 0)
            dropped_joints = info.get('dropped_joints', [])
            failure_prob = info.get('failure_probability', 0.0)

            # Log to SB3's logger (shows up in TensorBoard)
            if self.logger is not None:
                self.logger.record("curriculum/phase", current_phase)
                self.logger.record("curriculum/num_failed_joints", len(dropped_joints))
                self.logger.record("curriculum/failure_probability", failure_prob)

                # Log which specific joints are failing
                if len(dropped_joints) > 0:
                    self.logger.record("curriculum/has_failures", 1.0)
                else:
                    self.logger.record("curriculum/has_failures", 0.0)

            # Detect and log phase transitions
            if current_phase != self.last_logged_phase:
                self._log_phase_transition(current_phase)
                self.last_logged_phase = current_phase

            # Log per-episode statistics (when episode ends)
            for i, done in enumerate(self.locals.get('dones', [])):
                if done:
                    self.episode_count += 1

                    # Log episode-level curriculum stats
                    if wandb.run is not None:
                        wandb.log({
                            "curriculum/episode_phase": current_phase,
                            "curriculum/episode_failures": len(dropped_joints),
                            "curriculum/episode_count": self.episode_count
                        }, step=self.num_timesteps)

        return True

    def _log_phase_transition(self, new_phase: int):
        """Log phase transition with visual markers in W&B"""

        # Store transition timestep
        self.phase_transition_steps[new_phase] = self.num_timesteps

        # Log to W&B with annotation
        if wandb.run is not None:
            wandb.log({
                "curriculum/phase_transition": self.num_timesteps,
                "curriculum/new_phase": new_phase,
            }, step=self.num_timesteps)

            # Add to W&B summary for easy reference
            wandb.run.summary[f"phase_{new_phase}_start_step"] = self.num_timesteps
            wandb.run.summary[f"phase_{new_phase}_start_million"] = self.num_timesteps / 1e6

        # Print to console
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"📍 CURRICULUM PHASE TRANSITION")
            print(f"   New Phase: {new_phase}")
            print(f"   Timestep: {self.num_timesteps:,} ({self.num_timesteps/1e6:.1f}M)")
            print(f"{'='*70}\n")

    def _on_training_end(self) -> None:
        """Log final curriculum statistics"""
        if self.verbose > 0:
            print("\n" + "="*70)
            print("📊 CURRICULUM TRAINING SUMMARY")
            print("="*70)
            print(f"Total Episodes: {self.episode_count:,}")
            print(f"Phase Transitions:")
            for phase, step in sorted(self.phase_transition_steps.items()):
                print(f"  Phase {phase}: Started at {step:,} steps ({step/1e6:.1f}M)")
            print("="*70 + "\n")

        # Log summary to W&B
        if wandb.run is not None:
            wandb.run.summary["total_curriculum_episodes"] = self.episode_count
            wandb.run.summary["num_phase_transitions"] = len(self.phase_transition_steps)
