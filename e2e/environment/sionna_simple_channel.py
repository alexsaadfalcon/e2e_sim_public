# %%
# Import or install Sionna
import sionna.rt

# Other imports
import matplotlib.pyplot as plt
import numpy as np

import os
from tqdm import tqdm
no_preview = False # Toggle to False to use the preview widget

# Import relevant components from Sionna RT
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies

# %%
# Load integrated scene
scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile

# %%
if not no_preview:
    scene.preview();

# %%
# Only availabe if a preview is open
if not no_preview:
    scene.render(camera="preview", num_samples=512);

# %%
# Only availabe if a preview is open
if not no_preview:
    scene.render_to_file(camera="preview",
                         filename="scene.png",
                         resolution=[650,500]);

# %%
# Create new camera with different configuration
my_cam = Camera(position=[150,275,150], look_at=[30,70,28])
# Render scene with new camera*
scene.render(camera=my_cam, resolution=[650, 500], num_samples=512); # Increase num_samples to increase image quality

# %%
scene = load_scene(sionna.rt.scene.simple_street_canyon, merge_shapes=False)
scene.objects

# %%
floor = scene.get("floor")

# %%
print("Position (x,y,z) [m]: ", floor.position)
print("Orientation (alpha, beta, gamma) [rad]: ", floor.orientation)
print("Scaling: ", floor.scaling)

# %%
print("Velocity (x,y,z) [m/s]: ", floor.velocity)

# %%
floor.radio_material

# %%
scene.frequency = 31.5e9 # in Hz; implicitly updates RadioMaterials that implement frequency dependent properties
floor.radio_material # Note that the conductivity (sigma) changes automatically

# %%
scene = load_scene(sionna.rt.scene.munich, merge_shapes=True) # Merge shapes to speed-up computations

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=32,
                             num_cols=32,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

# Create transmitter
tx = Transmitter(name="tx",
                 position=[8.5,21,27],
                 display_radius=10)

# Add transmitter instance to scene
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx",
              position=[45,90,1.5],
              display_radius=10)

# Add receiver instance to scene
scene.add(rx)

tx.look_at(rx) # Transmitter points towards receiver
rx.look_at(tx)

# %%
scene.preview(show_devices=True, show_orientations=True)

# %%
# move receive across Munich square
rx.position += [-50, 0, 0]

# %%
# animation
# p_solver = PathSolver()
# os.makedirs('sionna_frames', exist_ok=True)
# for i in tqdm(range(100)):
#     paths = p_solver(scene=scene,
#                  max_depth=5,
#                  los=True,
#                  specular_reflection=True,
#                  diffuse_reflection=False,
#                  refraction=True,
#                  synthetic_array=True,
#                  seed=41)
#     scene.render_to_file(camera=my_cam,
#                          paths=paths,
#                          filename=f"sionna_frames/scene{i}.png",
#                          resolution=[650,500])
#     rx.position += [1, 0, 0]
# rx.position += [-100, 0, 0]
# from visualization.gif_utils import gif_folder
# gif_folder('sionna_frames/', 'scene', 40)
# exit()

# %%
# Instantiate a path solver
# The same path solver can be used with multiple scenes
p_solver = PathSolver()

# Compute propagation paths
paths = p_solver(scene=scene,
                 max_depth=5,
                 los=True,
                 specular_reflection=True,
                 diffuse_reflection=False,
                 refraction=True,
                 synthetic_array=False,
                 seed=41)

# %%
if no_preview:
    scene.render(camera=my_cam, paths=paths, clip_at=20);
else:
    scene.preview(paths=paths, clip_at=20);

# %%
a, tau = paths.cir(normalize_delays=True, out_type="numpy")
# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
print("Shape of a: ", a.shape)
# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
print("Shape of tau: ", tau.shape)

# %%
t = tau[0,0,0,0,:]/1e-9 # Scale to ns
a_abs = np.abs(a)[0,0,0,0,:,0]
a_max = np.max(a_abs)
# And plot the CIR
plt.figure()
plt.title("Channel impulse response")
plt.stem(t, a_abs)
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$");

# %%
# OFDM system parameters
num_subcarriers = 10000
subcarrier_spacing = 3e9 / num_subcarriers

# Compute frequencies of subcarriers relative to the carrier frequency
frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
frequencies = frequencies[:num_subcarriers//2]

# Compute channel frequency response
h_freq = paths.cfr(frequencies=frequencies,
                   normalize=True, # Normalize energy
                   normalize_delays=True,
                   out_type="numpy")
# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
print("Shape of h_freq: ", h_freq.shape)

# Plot absolute value
plt.figure()
plt.plot(np.abs(h_freq)[0,0,0,0,0,:]);
plt.xlabel("Subcarrier index");
plt.ylabel(r"|$h_\text{freq}$|");
plt.title("Channel frequency response");

# %%
taps = paths.taps(bandwidth=100e6, # Bandwidth to which the channel is low-pass filtered
                  l_min=-6,        # Smallest time lag
                  l_max=100,       # Largest time lag
                  sampling_frequency=None, # Sampling at Nyquist rate, i.e., 1/bandwidth
                  normalize=True,  # Normalize energy
                  normalize_delays=True,
                  out_type="numpy")
print("Shape of taps: ", taps.shape)

plt.figure()
plt.stem(np.arange(-6, 101), np.abs(taps)[0,0,0,0,0]);
plt.xlabel(r"Tap index $\ell$");
plt.ylabel(r"|$h[\ell]|$");
plt.title("Discrete channel taps");

all_s_pars = []
for _ in range(100):
    rx.position += [1, 0, 0]
    paths = p_solver(scene=scene,
                 max_depth=5,
                 los=True,
                 specular_reflection=True,
                 diffuse_reflection=False,
                 refraction=True,
                 synthetic_array=False,
                 seed=41)
    cfr = paths.cfr(frequencies=frequencies,
                   normalize=True, # Normalize energy
                   normalize_delays=True,
                   out_type="numpy")
    s_pars = cfr[0, :, 0, :, :, :]
    all_s_pars.append(s_pars)

all_s_pars = np.stack(all_s_pars, axis=0)

import os, pickle
this_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(this_dir, 'sionna_sims'), exist_ok=True)

out_fname = os.path.join(this_dir, 'sionna_sims', 'munich.pkl')
print('dumping to file', out_fname)
pickle.dump(all_s_pars, open(out_fname, 'wb'))
print('done dumping')
exit()

# %%
scene.get("tx").velocity = [10, 0, 0]

# Recompute propagation paths
paths_mob = p_solver(scene=scene,
                     max_depth=5,
                     los=True,
                     specular_reflection=True,
                     diffuse_reflection=False,
                     refraction=True,
                     synthetic_array=True,
                     seed=41)

# Compute CIR with time-evolution
num_time_steps=100
sampling_frequency = 1e4
a_mob, _ = paths_mob.cir(sampling_frequency=sampling_frequency,
                         num_time_steps=num_time_steps,
                         out_type="numpy")

# Inspect time-evolution of a single path coefficient
plt.figure()
plt.plot(np.arange(num_time_steps)/sampling_frequency*1000,
         a_mob[0,0,0,0,0].real);
plt.xlabel("Time [ms]");
plt.ylabel(r"$\Re\{a_0(t) \}$");
plt.title("Time-evolution of a path coefficient");

# %%
rm_solver = RadioMapSolver()

rm = rm_solver(scene=scene,
               max_depth=5,
               cell_size=[1,1],
               samples_per_tx=10**6)

# %%
if no_preview:
    scene.render(camera=my_cam, radio_map=rm);
else:
    scene.preview(radio_map=rm);
# %%
