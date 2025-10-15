# -*- coding: utf-8 -*-
"""
SionnaEnvironment: A class-based interface for Sionna RT radar simulation

This class provides a clean interface for:
- Creating and managing Sionna RT scenes with cars
- Positioning radar transmitters and receivers
- Placing cameras for visualization
- Computing and extracting channel responses
"""

import numpy as np
import matplotlib.pyplot as plt
import sionna.rt as rt
import drjit as dr
import mitsuba as mi
from typing import List, Tuple, Optional, Union


class SionnaEnvironment:
    """
    A class-based interface for Sionna RT radar simulation with cars.
    
    This class encapsulates the Sionna RT functionality and provides
    clean interfaces for radar positioning, camera placement, and
    channel response extraction.
    """
    
    def __init__(self, 
                 scene_name: str = "etoile",
                 frequency: float = 3.3e9,
                 num_cars: int = 10,
                 car_radius: float = 70.0,
                 car_center: Tuple[float, float, float] = (-127, 37, 15.5)):
        """
        Initialize the Sionna environment.
        
        Args:
            scene_name: Name of the scene to load (default: "etoile")
            frequency: Operating frequency in Hz (default: 40e9)
            num_cars: Number of cars to add to the scene (default: 10)
            car_radius: Radius of the circle for car placement (default: 100.0)
            car_center: Center point for car circle placement (default: (-127, 37, 1.5))
        """
        self.frequency = frequency
        self.num_cars = num_cars
        self.car_radius = car_radius
        self.car_center = car_center
        
        # Initialize scene
        self._setup_scene(scene_name)
        
        # Initialize radar components
        self.tx = None
        self.rx = None
        self.cars = []
        self.cars_positions = None
        self.theta_step = 0.
        
        # Initialize cameras
        self.cameras = {}
        
        # Initialize path solver
        self.path_solver = rt.PathSolver()
        self.paths = None
        
    def _setup_scene(self, scene_name: str):
        """Setup the base scene and antenna arrays."""
        print(f"Loading {scene_name} scene...")
        
        # Load scene
        if scene_name == "etoile":
            self.scene = rt.load_scene(merge_shapes=False)
            # self.scene = rt.load_scene(rt.scene.etoile, merge_shapes=False)
        elif scene_name == "san_francisco":
            self.scene = rt.load_scene(rt.scene.san_francisco)
        else:
            raise ValueError(f"Unknown scene: {scene_name}")
            
        self.scene.frequency = self.frequency
        
        # Configure antenna arrays
        pattern = "tr38901"
        self.scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                           vertical_spacing=0.5, horizontal_spacing=0.5,
                                           pattern="iso", polarization="V")
        
        self.scene.rx_array = rt.PlanarArray(num_rows=32, num_cols=32,
                                           vertical_spacing=0.5, horizontal_spacing=0.5,
                                           pattern="iso", polarization="V")
        
        print(f"Scene loaded with frequency: {float(self.scene.frequency.numpy())/1e9:.1f} GHz")
        print(f"Number of objects in scene: {len(self.scene.objects)}")
        
    def add_cars(self, 
                 car_material: Optional[rt.ITURadioMaterial] = None,
                 car_height_offset: float = 3.0):
        """
        Add cars to the scene in a circular pattern.
        
        Args:
            car_material: Material for the cars (default: red metal)
            car_height_offset: Height offset for receivers above cars
        """
        print(f"Adding {self.num_cars} cars to the scene...")
        
        # Create car material if not provided
        if car_material is None:
            car_material = rt.ITURadioMaterial("car-material",
                                             "metal",
                                             thickness=0.01,
                                             color=(0.8, 0.1, 0.1))
        
        # Create cars
        self.cars = [rt.SceneObject(fname=rt.scene.sphere,
                                  name=f"car-{i}",
                                  radio_material=car_material)
                    for i in range(self.num_cars)]
        
        # Add cars to scene
        self.scene.edit(add=self.cars)
        
        # Position cars in a circle
        self._position_cars()

        for i in range(self.num_cars):
            self.cars[i].scaling = 5.0
        
        # Store car top positions for easy access
        self.car_tops = []
        for i in range(self.num_cars):
            car_top = [self.cars_positions.x[i], 
                      self.cars_positions.y[i], 
                      self.cars_positions.z[i] + car_height_offset]
            self.car_tops.append(car_top)
        
        print(f"Added {self.num_cars} cars to the scene")
        
    def _position_cars(self):
        """Position cars in a circular pattern."""
        c = mi.Point3f(*self.car_center)
        r = self.car_radius
        thetas = dr.linspace(mi.Float, 0., dr.two_pi, self.num_cars, endpoint=False)
        self.cars_positions = c + mi.Point3f(dr.cos(thetas), dr.sin(thetas), 0.) * r
        
        # Orientations - cars look tangent to the circle
        d = dr.normalize(self.cars_positions - c)
        look_at_dirs = mi.Vector3f(d.y, -d.x, 0.)
        look_at_points = self.cars_positions + look_at_dirs
        
        # Set car positions and orientations
        for i in range(self.num_cars):
            self.cars[i].position = mi.Point3f(self.cars_positions.x[i], 
                                             self.cars_positions.y[i], 
                                             self.cars_positions.z[i])
            self.cars[i].look_at(mi.Point3f(look_at_points.x[i], 
                                          look_at_points.y[i], 
                                          look_at_points.z[i]))
    
    def step_cars(self, step_size: float = 1.0):
        """Step cars in a circular pattern."""
        c = mi.Point3f(*self.car_center)
        r = self.car_radius
        thetas = dr.linspace(mi.Float, 0., dr.two_pi, self.num_cars, endpoint=False)
        
        theta_step = step_size * np.pi / 180.
        self.theta_step += theta_step
        thetas += self.theta_step

        self.cars_positions = c + mi.Point3f(dr.cos(thetas), dr.sin(thetas), 0.) * r

        # Orientations - cars look tangent to the circle
        d = dr.normalize(self.cars_positions - c)
        look_at_dirs = mi.Vector3f(d.y, -d.x, 0.)
        look_at_points = self.cars_positions + look_at_dirs
        for i in range(self.num_cars):
            self.cars[i].position = mi.Point3f(self.cars_positions.x[i], 
                                             self.cars_positions.y[i], 
                                             self.cars_positions.z[i])
            self.cars[i].look_at(mi.Point3f(look_at_points.x[i], 
                                          look_at_points.y[i], 
                                          look_at_points.z[i]))


    def place_radar(self, 
                   tx_position: Union[List[float], Tuple[float, float, float], int],
                   rx_position: Union[List[float], Tuple[float, float, float], int, None] = None,
                   tx_look_at: Union[List[float], Tuple[float, float, float], None] = None,
                   rx_look_at: Union[List[float], Tuple[float, float, float], None] = None):
        """
        Place radar transmitter and receiver.
        
        Args:
            tx_position: TX position as [x, y, z] or car index (int)
            rx_position: RX position as [x, y, z] or car index (int). 
                        If None, uses same as TX position.
        """
        # Handle TX position
        if isinstance(tx_position, int):
            if tx_position >= len(self.car_tops):
                raise ValueError(f"Car index {tx_position} out of range (0-{len(self.car_tops)-1})")
            tx_pos = self.car_tops[tx_position]
        else:
            tx_pos = list(tx_position)
            
        # Handle RX position
        if rx_position is None:
            rx_pos = tx_pos
        elif isinstance(rx_position, int):
            if rx_position >= len(self.car_tops):
                raise ValueError(f"Car index {rx_position} out of range (0-{len(self.car_tops)-1})")
            rx_pos = self.car_tops[rx_position]
        else:
            rx_pos = list(rx_position)
        
        # Remove existing TX/RX if they exist
        if self.tx is not None:
            self.scene.remove(self.tx)
        if self.rx is not None:
            self.scene.remove(self.rx)
            
        # Add new TX and RX
        self.tx = rt.Transmitter(name="tx", position=tx_pos, look_at=tx_look_at)
        self.rx = rt.Receiver(name="rx", position=rx_pos, look_at=rx_look_at)
        
        self.scene.add(self.tx)
        self.scene.add(self.rx)
        
        print(f"Radar placed - TX: {tx_pos}, RX: {rx_pos}")
        print(f"Distance: {np.linalg.norm(np.array(rx_pos) - np.array(tx_pos)):.1f} m")
        
    def add_camera(self, 
                  name: str,
                  position: Union[List[float], Tuple[float, float, float]],
                  look_at: Union[List[float], Tuple[float, float, float]],
                  resolution: Tuple[int, int] = (800, 600)):
        """
        Add a camera to the scene.
        
        Args:
            name: Name for the camera
            position: Camera position [x, y, z]
            look_at: Point to look at [x, y, z]
            resolution: Image resolution (width, height)
        """
        camera = rt.Camera(position=list(position), look_at=list(look_at))
        self.cameras[name] = {
            'camera': camera,
            'resolution': resolution
        }
        print(f"Added camera '{name}' at {position} looking at {look_at}")
        
    def add_top_down_camera(self, 
                           name: str = "top_down",
                           height: float = 500.0,
                           center: Optional[Tuple[float, float, float]] = None,
                           resolution: Tuple[int, int] = (800, 600)):
        """Add a top-down camera looking down at the scene."""
        if center is None:
            center = (*self.car_center[:2], 0)
            
        position = [center[0], center[1], height]
        look_at = [center[0], center[1], center[2]]
        
        self.add_camera(name, position, look_at, resolution)
        
    def add_car_camera(self, 
                      name: str = "car_level",
                      car_index: int = 0,
                      offset: Tuple[float, float, float] = (20, 20, 5),
                      resolution: Tuple[int, int] = (800, 600)):
        """Add a camera positioned near a specific car."""
        if car_index >= len(self.car_tops):
            raise ValueError(f"Car index {car_index} out of range (0-{len(self.car_tops)-1})")
            
        car_pos = self.car_tops[car_index]
        position = [car_pos[0] + offset[0], car_pos[1] + offset[1], car_pos[2] + offset[2]]
        look_at = car_pos
        
        self.add_camera(name, position, look_at, resolution)
        
    def render_scene(self, 
                    camera_name: str,
                    show_paths: bool = False,
                    title: Optional[str] = None):
        """
        Render the scene from a specific camera.
        
        Args:
            camera_name: Name of the camera to use
            show_paths: Whether to show radio propagation paths
            title: Title for the plot
        """
        if camera_name not in self.cameras:
            raise ValueError(f"Camera '{camera_name}' not found")
            
        camera_info = self.cameras[camera_name]
        camera = camera_info['camera']
        resolution = camera_info['resolution']
        
        try:
            if show_paths and self.paths is not None:
                img = self.scene.render(camera=camera, paths=self.paths, resolution=resolution)
            else:
                img = self.scene.render(camera=camera, resolution=resolution)
                
            if title:
                plt.title(title)
            plt.show()
            
        except Exception as e:
            print(f"Error rendering scene with camera '{camera_name}': {e}")
            
    def compute_paths(self, 
                     max_depth: int = 5,
                     los: bool = True,
                     specular_reflection: bool = True,
                     synthetic_array: bool = False):
        """
        Compute radio propagation paths.
        
        Args:
            max_depth: Maximum number of reflections
            los: Include line-of-sight paths
            specular_reflection: Include specular reflections
            synthetic_array: Use synthetic array processing
        """
        print("Computing radio propagation paths...")
        
        self.paths = self.path_solver(self.scene, 
                                    max_depth=max_depth,
                                    los=los,
                                    specular_reflection=specular_reflection,
                                    synthetic_array=synthetic_array)
        
        # Extract basic path information
        a, tau = self.paths.cir(out_type="numpy")
        print(f"CIR shape: a={a.shape}, tau={tau.shape}")
        print(f"Number of paths found: {a.shape[4]}")
        
        return self.paths
        
    def get_channel_response(self, 
                           antenna_index: int = 0,
                           max_paths: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract channel impulse response for a specific antenna.
        
        Args:
            antenna_index: Index of the antenna to analyze
            max_paths: Maximum number of paths to return
            
        Returns:
            Tuple of (amplitudes, delays_ns, powers_db)
        """
        if self.paths is None:
            raise ValueError("No paths computed. Call compute_paths() first.")
            
        # Extract channel data
        a, tau = self.paths.cir(out_type="numpy")
        
        if a.shape[4] == 0:
            print("No paths found.")
            return np.array([0.0]), np.array([0.0]), np.array([-200.0])
        
        # Get path parameters for specified antenna
        a_ant = a[0, antenna_index, 0, 0, :, 0]  # [rx, rx_ant, tx, tx_ant, path, time]
        tau_ns = tau[0, antenna_index, 0, 0, :] * 1e9
        power_db = 20 * np.log10(np.abs(a_ant) + 1e-12)
        
        # Sort by power and return top paths
        sort_idx = np.argsort(power_db)[::-1]
        top_paths = sort_idx[:max_paths]
        
        return a_ant[top_paths], tau_ns[top_paths], power_db[top_paths]
        
    def plot_channel_response(self, 
                            antenna_indices: List[int] = [0, 8, 16, 24, 32],
                            max_paths: int = 5):
        """
        Plot channel impulse response for multiple antennas.
        
        Args:
            antenna_indices: List of antenna indices to plot
            max_paths: Maximum number of paths to show
        """
        if self.paths is None:
            raise ValueError("No paths computed. Call compute_paths() first.")
            
        # Extract channel data
        a, tau = self.paths.cir(out_type="numpy")
        
        if a.shape[4] == 0:
            print("No paths found for plotting.")
            return
            
        plt.figure(figsize=(14, 5))
        
        # Plot 1: Power delay profile for first antenna
        plt.subplot(1, 3, 1)
        a_ant0 = a[0, 0, 0, 0, :, 0]
        tau_ns = tau[0, 0, 0, 0, :] * 1e9
        power_db = 20 * np.log10(np.abs(a_ant0) + 1e-12)
        
        plt.stem(tau_ns, power_db, basefmt=' ')
        plt.xlabel('Delay [ns]')
        plt.ylabel('Power [dB]')
        plt.title('Power Delay Profile (Antenna 0)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Overlay of multiple antennas
        plt.subplot(1, 3, 2)
        for ant_idx in antenna_indices:
            if ant_idx < a.shape[1]:  # Check if antenna index is valid
                a_ant = a[0, ant_idx, 0, 0, :, 0]
                power = 20 * np.log10(np.abs(a_ant) + 1e-12)
                plt.plot(tau_ns, power, 'o-', alpha=0.7, label=f'Ant {ant_idx}', markersize=4)
        plt.xlabel('Delay [ns]')
        plt.ylabel('Power [dB]')
        plt.title('Power Delay Profile (Select Antennas)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Received power across array
        plt.subplot(1, 3, 3)
        array_powers = np.zeros(64)
        for ant_idx in range(min(64, a.shape[1])):
            a_ant = a[0, ant_idx, 0, 0, :, 0]
            array_powers[ant_idx] = 20 * np.log10(np.sum(np.abs(a_ant)) + 1e-12)
        array_powers_2d = array_powers.reshape(8, 8)
        im = plt.imshow(array_powers_2d, cmap='hot', interpolation='nearest')
        plt.colorbar(im, label='Power [dB]')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.title('Total Received Power (8x8 Array)')
        
        plt.tight_layout()
        plt.show()
        
    def get_scene_info(self) -> dict:
        """Get information about the current scene."""
        return {
            'frequency_ghz': float(self.scene.frequency.numpy()) / 1e9,
            'num_objects': len(self.scene.objects),
            'num_cars': len(self.cars),
            'tx_position': self.tx.position if self.tx else None,
            'rx_position': self.rx.position if self.rx else None,
            'num_cameras': len(self.cameras),
            'paths_computed': self.paths is not None
        }

    def get_S_pars(self, freqs):
        cfr = self.paths.cfr(freqs, out_type='numpy')
        s_pars = cfr[0, :, 0, :, :, :]
        return s_pars

