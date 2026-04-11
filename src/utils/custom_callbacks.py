from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomMetricsCallback(BaseCallback):
    """
    Custom callback for logging our specific metrics to W&B
    """
    def __init__(self, verbose=0):
        super(CustomMetricsCallback, self).__init__(verbose)
        self.episode_velocities = []
        self.episode_distances = []
        
    def _on_step(self) -> bool:
        # Check if any episodes finished
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                
                # Log custom metrics if available (check each one)
                if 'custom_metrics/instant_velocity' in info:
                    self.logger.record('custom/final_velocity', info['custom_metrics/instant_velocity'])
                
                if 'custom_metrics/distance_traveled' in info:
                    self.logger.record('custom/total_distance', info['custom_metrics/distance_traveled'])
                
                if 'custom_metrics/is_walking_slowly' in info:
                    self.logger.record('custom/is_slow_walking', float(info['custom_metrics/is_walking_slowly']))
                
                # Only log if it exists
                if 'custom_metrics/speed_penalty' in info:
                    self.logger.record('custom/speed_penalty', info['custom_metrics/speed_penalty'])
                
                # New metrics from updated wrapper
                if 'custom_metrics/height' in info:
                    self.logger.record('custom/robot_height', info['custom_metrics/height'])
                    
                if 'custom_metrics/gait_quality' in info:
                    self.logger.record('custom/gait_quality', info['custom_metrics/gait_quality'])
                
                # Track for averaging
                if 'custom_metrics/instant_velocity' in info:
                    self.episode_velocities.append(info['custom_metrics/instant_velocity'])
                if 'custom_metrics/distance_traveled' in info:
                    self.episode_distances.append(info['custom_metrics/distance_traveled'])
        
        # Log averages every 100 episodes
        if len(self.episode_velocities) >= 100:
            self.logger.record('custom/avg_velocity_100ep', np.mean(self.episode_velocities))
            self.logger.record('custom/avg_distance_100ep', np.mean(self.episode_distances))
            self.episode_velocities = []
            self.episode_distances = []
            
        return True