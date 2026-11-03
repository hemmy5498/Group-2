import numpy as np
import pandas as pd

def extract_kinematic_features(df, user_id):
    """Calculates velocity, acceleration, and jerk."""
    dt = df['timestamp'].diff().replace(0, np.nan)
    distance = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2)
    velocity = distance / dt
    acceleration = velocity.diff() / dt
    jerk = acceleration.diff() / dt
    return {'user_id': user_id, 'mean_velocity': np.nan_to_num(velocity.mean())}
