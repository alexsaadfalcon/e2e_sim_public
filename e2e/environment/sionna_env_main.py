# %% setup class
from sionna_env import SionnaEnvironment

import os
import pickle
import numpy as np

freqs = np.linspace(2.7e9, 3.3e9, 1000)
freqs -= 3e9

"""
Reproduce the results from sionna_channel.py using the class-based interface.
"""
print("Reproducing sionna_channel.py results with class-based interface...")

# Create environment
env = SionnaEnvironment(frequency=3.3e9, num_cars=10)

# Add cars
env.add_cars()

# Place radar on car 5 (same as in the modified sionna_channel.py)
arch_pos = [-100, 37, 10]
arch_look_at = [-90, 37, 10]
arch_pos_tx = [-100, 37, 15]
arch_look_at_tx = [-90, 37, 15]
env.place_radar(tx_position=arch_pos_tx, rx_position=arch_pos, tx_look_at=arch_look_at_tx, rx_look_at=arch_look_at)

# %% Preview
env.scene.preview(show_devices=True, show_orientations=True)

# %% Step cars
env.step_cars(step_size=10.)
env.compute_paths(max_depth=5)
s_pars = env.get_S_pars(freqs)
print(s_pars.shape)

# %% Step cars in a loop and dump S-parameters
all_s_pars = []
for _ in range(10):
    env.step_cars(10.0)
    env.compute_paths(max_depth=50, synthetic_array=True)
    s_pars = env.get_S_pars(freqs)
    all_s_pars.append(s_pars)

all_s_pars = np.stack(all_s_pars, axis=0)
os.makedirs('sionna_sims', exist_ok=True)

pickle.dump(all_s_pars, open('sionna_sims/etoile.pkl', 'wb'))

# %% Add cameras

# Add cameras
env.add_top_down_camera(height=500)
env.add_car_camera(car_index=5, offset=(20, 20, 5))

# Render initial views
print("\nRendering initial views...")
env.render_scene("top_down", title="Top-down view of Etoile scene with cars")
env.render_scene("car_level", title="Car-level view of the scene")

# Compute paths
env.compute_paths(max_depth=5)

# Plot channel response
print("\nPlotting channel response...")
env.plot_channel_response()

# Render final views with paths
print("\nRendering final views with paths...")
env.render_scene("top_down", show_paths=True, title="Top-down view with radio propagation paths")

# Print scene info
info = env.get_scene_info()
print(f"\nScene information:")
for key, value in info.items():
    print(f"  {key}: {value}")
# %%
