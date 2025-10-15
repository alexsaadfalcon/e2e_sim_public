# %%
"""
Minimal Sionna example: 1 TX to 8x8 UPA using Etoile scene with cars
"""
import numpy as np
import matplotlib.pyplot as plt
import sionna.rt as rt
import drjit as dr
import mitsuba as mi

# Load Etoile scene
scene = rt.load_scene(rt.scene.etoile, merge_shapes=True)
scene.frequency = 3.5e9  # 3.5 GHz
scene.frequency = 40e9

# Add cars to the scene
print("Adding cars to the scene...")
num_cars = 10

# Radio material for the cars
car_material = rt.ITURadioMaterial("car-material",
                                  "metal",
                                  thickness=0.01,
                                  color=(0.8, 0.1, 0.1))

# Create cars
cars = [rt.SceneObject(fname=rt.scene.low_poly_car,
                      name=f"car-{i}",
                      radio_material=car_material)
        for i in range(num_cars)]

# Add cars to scene
scene.edit(add=cars)

# Position cars in a circle around the central monument
c = mi.Point3f(-127, 37, 1.5)  # Center of the circle
r = 100  # Radius of the circle
thetas = dr.linspace(mi.Float, 0., dr.two_pi, num_cars, endpoint=False)
cars_positions = c + mi.Point3f(dr.cos(thetas), dr.sin(thetas), 0.)*r

# Orientations - cars look tangent to the circle
d = dr.normalize(cars_positions - c)
look_at_dirs = mi.Vector3f(d.y, -d.x, 0.)
look_at_points = cars_positions + look_at_dirs

# Set car positions and orientations
for i in range(num_cars):
    cars[i].position = mi.Point3f(cars_positions.x[i], cars_positions.y[i], cars_positions.z[i])
    cars[i].look_at(mi.Point3f(look_at_points.x[i], look_at_points.y[i], look_at_points.z[i]))

print(f"Added {num_cars} cars to the scene")

# Configure arrays
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                              vertical_spacing=0.5, horizontal_spacing=0.5,
                              pattern="tr38901", polarization="V")

scene.rx_array = rt.PlanarArray(num_rows=8, num_cols=8,
                              vertical_spacing=0.5, horizontal_spacing=0.5,
                              pattern="tr38901", polarization="V")

# Add transmitter and receiver at realistic positions
car_idx = 0
car_top = [
    cars_positions.x[car_idx], 
    cars_positions.y[car_idx], 
    cars_positions.z[car_idx] + 3
]
arr_look_at = [
    cars_positions.x[car_idx], 
    cars_positions.y[car_idx] - 5, 
    cars_positions.z[car_idx] + 3,
]
arch_pos = [-100, 37, 3]
arch_look_at = [-90, 37, 3]
tx = rt.Transmitter(name="tx", position=arch_pos, look_at=arch_look_at)  # On top of car
rx = rt.Receiver(name="rx", position=arch_pos, look_at=arch_look_at)  # On top of car
scene.add(tx)
scene.add(rx)

# %% Preview scene
scene.preview(show_devices=True, show_orientations=True)


# %%

print(f"TX position: {tx.position}")
print(f"RX position: {rx.position}")
print(f"Distance: {np.linalg.norm(np.array(rx.position) - np.array(tx.position)):.1f} m")

# Print scene information
print(f"\nScene frequency: {float(scene.frequency.numpy())/1e9:.1f} GHz")
print(f"Number of objects in scene: {len(scene.objects)}")

# Add cameras to visualize the scene with cars
print("\nAdding cameras for visualization...")

# Top-down camera looking down at the scene from above
top_down_pos = [-127, 37, 500]  # High above the center
top_down_look = [-127, 37, 0]   # Looking down at the center
top_down_camera = rt.Camera(position=top_down_pos, look_at=top_down_look)

# Car-level camera positioned near one of the cars
car_camera_pos = [cars_positions.x[car_idx], cars_positions.y[car_idx]-100, cars_positions.z[car_idx]+100]
car_camera_look = [cars_positions.x[car_idx], cars_positions.y[car_idx], cars_positions.z[car_idx]]
car_camera = rt.Camera(position=car_camera_pos, look_at=car_camera_look)

print(f"Created cameras: Top-down camera at {top_down_pos}, Car camera at {car_camera_pos}")

# Render scene from different camera perspectives
print("\nRendering scene from different perspectives...")
try:
    # Render top-down view
    print("Rendering top-down view...")
    img = scene.render(camera=top_down_camera, resolution=[800, 600])
    plt.title("Top-down view of Etoile scene with cars")
    plt.show()
    
    # Render car-level view
    print("Rendering car-level view...")
    img = scene.render(camera=car_camera, resolution=[800, 600])
    plt.title("Car-level view of the scene")
    plt.show()
    
except Exception as e:
    print(f"Error rendering scene: {e}")
    print("Continuing with path computation...")

# Compute paths
print("\nComputing radio propagation paths...")
p_solver = rt.PathSolver()
paths = p_solver(scene, max_depth=5, los=False,
                specular_reflection=True, synthetic_array=False)

# Extract channel data
a, tau = paths.cir(out_type="numpy")
print(f"\nCIR shape: a={a.shape}, tau={tau.shape}")
print(f"Number of paths found: {a.shape[4]}")

# If no paths found, try to get more information
if a.shape[4] == 0:
    print("\nNo paths found. This could be due to:")
    print("- Transmitter/receiver positions outside scene bounds")
    print("- No line-of-sight and insufficient reflections")
    print("- Scene objects blocking all paths")
    print("- Path solver parameters too restrictive")
    
    # Try to get path information directly
    try:
        path_info = paths.paths
        print(f"Path info available: {path_info is not None}")
        if path_info is not None:
            print(f"Path info shape: {path_info.shape if hasattr(path_info, 'shape') else 'No shape'}")
    except Exception as e:
        print(f"Could not access path info: {e}")
    
    # Try alternative positions
    print("\nTrying alternative positions...")
    scene.remove(tx)
    scene.remove(rx)
    
    # Try positions that are more likely to have line-of-sight
    tx_alt = rt.Transmitter(name="tx_alt", position=[0, 0, 50])  # Higher altitude
    rx_alt = rt.Receiver(name="rx_alt", position=[100, 0, 50])   # Higher altitude, further away
    scene.add(tx_alt)
    scene.add(rx_alt)
    
    print(f"Alternative TX position: {tx_alt.position}")
    print(f"Alternative RX position: {rx_alt.position}")
    
    # Try path computation with alternative positions
    paths_alt = rt.PathSolver()(scene, max_depth=3, los=True,
                               specular_reflection=True, synthetic_array=False)
    
    a_alt, tau_alt = paths_alt.cir(out_type="numpy")
    print(f"Alternative CIR shape: a={a_alt.shape}, tau={tau_alt.shape}")
    print(f"Alternative paths found: {a_alt.shape[4]}")
    
    if a_alt.shape[4] > 0:
        print("Found paths with alternative positions! Using these instead.")
        a, tau = a_alt, tau_alt
        paths = paths_alt
        tx, rx = tx_alt, rx_alt

# Get path parameters for first antenna
if a.shape[4] > 0:
    a_ant0 = a[0, 0, 0, 0, :, 0]  # [rx, rx_ant, tx, tx_ant, path, time]
    tau_ns = tau[0, 0, 0, 0, :] * 1e9
    power_db = 20 * np.log10(np.abs(a_ant0) + 1e-12)

    # Sort by power
    sort_idx = np.argsort(power_db)[::-1]
    print(f"\nTop 5 paths:")
    print(f"{'Path':<6} {'Delay[ns]':<12} {'Power[dB]':<12}")
    for i in sort_idx[:50]:
        print(f"{i:<6} {tau_ns[i]:<12.2f} {power_db[i]:<12.2f}")
else:
    print("\nNo paths found even with alternative positions.")
    print("This might indicate an issue with the scene setup or Sionna RT configuration.")
    # Create dummy data for plotting
    a_ant0 = np.array([0.0])
    tau_ns = np.array([0.0])
    power_db = np.array([-200.0])

# Plot results
plt.figure(figsize=(14, 5))

# Plot 1: Power delay profile
plt.subplot(1, 3, 1)
plt.stem(tau_ns, power_db, basefmt=' ')
plt.xlabel('Delay [ns]')
plt.ylabel('Power [dB]')
plt.title('Power Delay Profile (Antenna 0)')
plt.xlim([0, max(tau_ns)])
plt.grid(True, alpha=0.3)

# Plot 2: Overlay of multiple antennas
plt.subplot(1, 3, 2)
if a.shape[4] > 0:
    for ant_idx in [0, 8, 16, 24, 32]:  # Corner and center antennas
        a_ant = a[0, ant_idx, 0, 0, :, 0]
        power = 20 * np.log10(np.abs(a_ant) + 1e-12)
        plt.plot(tau_ns, power, 'o-', alpha=0.7, label=f'Ant {ant_idx}', markersize=4)
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No paths found', ha='center', va='center', transform=plt.gca().transAxes)
plt.xlabel('Delay [ns]')
plt.ylabel('Power [dB]')
plt.xlim([0, max(tau_ns)])
plt.title('Power Delay Profile (Select Antennas)')
plt.grid(True, alpha=0.3)

# Plot 3: Received power across array
plt.subplot(1, 3, 3)
if a.shape[4] > 0:
    array_powers = np.zeros(64)
    for ant_idx in range(64):
        a_ant = a[0, ant_idx, 0, 0, :, 0]
        array_powers[ant_idx] = 20 * np.log10(np.sum(np.abs(a_ant)) + 1e-12)
    array_powers_2d = array_powers.reshape(8, 8)
    im = plt.imshow(array_powers_2d, cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Power [dB]')
else:
    # Show empty array
    array_powers_2d = np.full((8, 8), -200)
    im = plt.imshow(array_powers_2d, cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Power [dB]')
    plt.text(4, 4, 'No paths', ha='center', va='center', color='white', fontsize=12)
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Total Received Power (8x8 Array)')

plt.tight_layout()
plt.show()

# Final rendering with paths visible
print(f"\nRendering final view with radio propagation paths...")
try:
    # Render top-down view with paths
    img = scene.render(camera=top_down_camera, paths=paths, resolution=[800, 600])
    plt.title("Top-down view with radio propagation paths")
    plt.show()
    
    # Render car-level view with paths
    # img = scene.render(camera=car_camera, paths=paths, resolution=[800, 600])
    # plt.title("Car-level view with radio propagation paths")
    # plt.show()
    
except Exception as e:
    print(f"Error rendering final views: {e}")

print(f"\nDone!")
# %%
