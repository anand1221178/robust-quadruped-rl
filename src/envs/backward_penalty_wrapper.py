
import gymnasium as gym

class BackwardPenaltyWrapper(gym.Wrapper):
    """
    A wrapper to apply penalties for backward movement and optionally for staying stationary.
    It reads the velocity from the info dict (provided by SuccessRewardWrapper)
    and applies penalties to discourage undesired behaviors.
    """
    def __init__(self, env: gym.Env,
                 penalty_multiplier: float = 5.0,
                 stationary_penalty: float = 0.0,
                 stationary_threshold: float = 0.05):
        super().__init__(env)
        self.penalty_multiplier = penalty_multiplier
        self.stationary_penalty = stationary_penalty
        self.stationary_threshold = stationary_threshold

        print(f"🔥 BackwardPenaltyWrapper ACTIVE:")
        print(f"   - Backward penalty: x{self.penalty_multiplier}")
        if self.stationary_penalty != 0.0:
            print(f"   - Stationary penalty: {self.stationary_penalty} per step (|v| < {self.stationary_threshold})")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # SuccessRewardWrapper puts current_velocity in the info dict
        velocity = info.get('current_velocity', 0.0)

        # Apply penalties based on motion state
        if velocity < -self.stationary_threshold:
            # Backward motion - apply multiplier
            reward *= self.penalty_multiplier
            info['backward_penalty_applied'] = True
            info['motion_state'] = 'backward'

        elif abs(velocity) < self.stationary_threshold and self.stationary_penalty != 0.0:
            # Nearly stationary - apply fixed penalty if configured
            reward += self.stationary_penalty  # Add negative value
            info['stationary_penalty_applied'] = True
            info['motion_state'] = 'stationary'

        else:
            # Forward motion - no penalty
            info['motion_state'] = 'forward'

        return obs, reward, terminated, truncated, info
