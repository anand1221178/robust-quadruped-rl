
import gymnasium as gym

class BackwardPenaltyWrapper(gym.Wrapper):
    """
    A wrapper to apply a harsh penalty for backward movement.
    It reads the velocity from the info dict (provided by SuccessRewardWrapper)
    and applies a multiplier to any negative reward, discouraging backward walking.
    """
    def __init__(self, env: gym.Env, penalty_multiplier: float = 5.0):
        super().__init__(env)
        self.penalty_multiplier = penalty_multiplier
        print(f"🔥 BackwardPenaltyWrapper ACTIVE: Applying x{self.penalty_multiplier} penalty for backward movement.")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # SuccessRewardWrapper puts current_velocity in the info dict
        velocity = info.get('current_velocity', 0.0)

        # If velocity is negative, the reward from SuccessRewardWrapper will also be negative.
        # We make it even more negative to create a harsh penalty.
        if velocity < 0:
            reward *= self.penalty_multiplier
            info['backward_penalty_applied'] = True

        return obs, reward, terminated, truncated, info
