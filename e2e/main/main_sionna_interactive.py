import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

from e2e.environment.sionna_iterator import SionnaMunichIterator
from e2e.simulation import Simulation
from e2e.blocks import \
    SionnaEnvironmentBlock, \
    RFFEBlock, \
    InterconnectBlock, \
    AFEBlock, \
    AdaOjaBlock, \
    FFTBlock, \
    RangeAzBlock, \
    RangeElBlock, \
    SubspaceErrorBlock


class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sionna Interactive Simulation Pipeline")
        self.root.geometry("1200x800")
        
        # Simulation parameters
        self.N_RX_X = 32
        self.N_RX_Y = 32
        self.N_RX = self.N_RX_X * self.N_RX_Y
        self.N_TX = 1
        self.N_FREQS = 5000
        self.freqs = np.linspace(28.5e9, 31.5e9, self.N_FREQS)
        self.d = 16
        
        # Initialize blocks
        self.environment_block = SionnaEnvironmentBlock('munich')
        self.downstream_blocks = [
            FFTBlock(),
            RangeAzBlock(),
            RangeElBlock(),
            SubspaceErrorBlock(),
        ]
        self.subspace_block = AdaOjaBlock(self.N_RX, self.d)
        
        # Block states
        self.circuit_block = None
        self.interconnect_block = None
        self.afe_block = None
        
        # Initialize simulation object
        self.sim = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Simulation Pipeline Control", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create 2x2 grid
        # Top row
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Top left: Controls
        self.controls_frame = ttk.LabelFrame(top_frame, text="Pipeline Controls", padding=10)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Stage 1: Circuit Block
        stage1_frame = ttk.Frame(self.controls_frame)
        stage1_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(stage1_frame, text="1. Circuit Block (RFFE):", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        self.circuit_var = tk.StringVar(value="None")
        circuit_combo = ttk.Combobox(stage1_frame, textvariable=self.circuit_var, 
                                   values=["None", "Low Scaling (1e-5)", "High Scaling (1e-3)"], 
                                   state="readonly", width=20)
        circuit_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Stage 2: Interconnect Block
        stage2_frame = ttk.Frame(self.controls_frame)
        stage2_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(stage2_frame, text="2. Interconnect Block:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        self.interconnect_var = tk.BooleanVar(value=False)
        interconnect_check = ttk.Checkbutton(stage2_frame, text="Enable Interconnect", 
                                           variable=self.interconnect_var)
        interconnect_check.pack(side=tk.LEFT, padx=(10, 0))
        
        # Stage 3: AFE Block
        stage3_frame = ttk.Frame(self.controls_frame)
        stage3_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(stage3_frame, text="3. AFE Block:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        self.afe_var = tk.BooleanVar(value=False)
        afe_check = ttk.Checkbutton(stage3_frame, text="Enable AFE", 
                                   variable=self.afe_var)
        afe_check.pack(side=tk.LEFT, padx=(10, 0))
        
        # Control buttons
        control_frame = ttk.Frame(self.controls_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(control_frame, text="Run Simulation", 
                  command=self.run_simulation).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT)
        
        # Top right: Scene image
        self.scene_frame = ttk.LabelFrame(top_frame, text="Scene View", padding=10)
        self.scene_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Bottom row
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bottom left: FFT Linear
        self.fft_linear_frame = ttk.LabelFrame(bottom_frame, text="FFT - Linear Scale", padding=10)
        self.fft_linear_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Bottom right: FFT Logarithmic
        self.fft_log_frame = ttk.LabelFrame(bottom_frame, text="FFT - Logarithmic Scale", padding=10)
        self.fft_log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
    def update_blocks(self):
        """Update block configurations based on GUI selections"""
        # Circuit block
        circuit_choice = self.circuit_var.get()
        if circuit_choice == "None":
            self.circuit_block = None
        elif circuit_choice == "Low Scaling (1e-5)":
            self.circuit_block = RFFEBlock(n=self.N_RX * self.N_TX, chirp_dur=10e-9, signal_scaling=1e-5)
        elif circuit_choice == "High Scaling (1e-3)":
            self.circuit_block = RFFEBlock(n=self.N_RX * self.N_TX, chirp_dur=10e-9, signal_scaling=1e-3)
        
        # Interconnect block
        if self.interconnect_var.get():
            self.interconnect_block = InterconnectBlock(case='case3')
        else:
            self.interconnect_block = None
        
        # AFE block
        if self.afe_var.get():
            self.afe_block = AFEBlock()
        else:
            self.afe_block = None
    
    def initialize_simulation(self):
        """Initialize the simulation object if it doesn't exist"""
        if self.sim is None:
            self.sim = Simulation(
                self.environment_block,
                self.downstream_blocks,
                self.d,
                self.circuit_block,
                self.interconnect_block,
                self.afe_block,
                self.subspace_block,
            )
        else:
            # Update existing simulation blocks
            self.sim.environment_block = self.environment_block
            self.sim.circuit_block = self.circuit_block
            self.sim.interconnect_block = self.interconnect_block
            self.sim.afe_block = self.afe_block
            self.sim.subspace_block = self.subspace_block
    
    def run_simulation(self):
        """Run the simulation with current block configurations"""
        try:
            # Update block configurations
            self.update_blocks()
            
            # Initialize or update simulation
            self.initialize_simulation()
            
            # Run simulation step by step to avoid resetting
            for _ in range(10):
                self.sim.step()
            print('sim step', self.sim.environment_block.frame_counter)
            self.sim.feed_forward()
            
            # Get outputs
            outputs = self.sim.get_outputs()
            
            # Display results
            self.display_results(outputs)
            
        except Exception as e:
            messagebox.showerror("Simulation Error", f"An error occurred: {str(e)}")
    
    def display_results(self, outputs):
        """Display simulation results in the GUI"""
        # Clear previous results
        for widget in self.scene_frame.winfo_children():
            widget.destroy()
        for widget in self.fft_linear_frame.winfo_children():
            widget.destroy()
        for widget in self.fft_log_frame.winfo_children():
            widget.destroy()
        
        # Display scene image
        if self.sim and self.sim.environment_block:
            frame_counter = self.sim.environment_block.frame_counter
            scene_path = f"v1/environment/sionna_frames/scene{frame_counter//2}.png"
            try:
                from PIL import Image, ImageTk
                import os
                
                if os.path.exists(scene_path):
                    # Load and display image
                    img = Image.open(scene_path)
                    
                    # Get the frame size to scale image to fit the subwindow
                    self.scene_frame.update_idletasks()
                    frame_width = self.scene_frame.winfo_width()
                    frame_height = self.scene_frame.winfo_height()
                    
                    # If frame size is not available yet, use a reasonable default
                    if frame_width <= 1 or frame_height <= 1:
                        frame_width = 400
                        frame_height = 300
                    
                    # Calculate scaling to fit within the frame while maintaining aspect ratio
                    img_width, img_height = img.size
                    scale_w = frame_width / img_width
                    scale_h = frame_height / img_height
                    scale = min(scale_w, scale_h)  # Use min to fit within the frame
                    
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    
                    # Ensure the image doesn't exceed frame dimensions
                    new_width = min(new_width, frame_width)
                    new_height = min(new_height, frame_height)
                    
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    scene_label = ttk.Label(self.scene_frame, image=photo)
                    scene_label.image = photo  # Keep a reference
                    scene_label.pack(fill=tk.BOTH, expand=True)
                else:
                    ttk.Label(self.scene_frame, text=f"Scene {frame_counter//2} not found").pack(expand=True)
            except ImportError:
                ttk.Label(self.scene_frame, text="PIL not available for image display").pack(expand=True)
            except Exception as e:
                ttk.Label(self.scene_frame, text=f"Error loading scene: {str(e)}").pack(expand=True)
        
        # Plot FFT results
        if 'fft' in outputs and outputs['fft']:
            _fft = outputs['fft'][-1]
            _fft = _fft / torch.max(torch.abs(_fft))
            fft_energy = 20 * torch.log10(torch.abs(_fft)).T.cpu()
            
            # Linear scale FFT (bottom left)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            im1 = ax1.imshow(torch.abs(_fft).T.cpu(), aspect='auto')
            ax1.set_title('FFT - Linear Scale')
            ax1.set_xlabel('Frequency Bin')
            ax1.set_ylabel('Time Bin')
            plt.colorbar(im1, ax=ax1, label='Magnitude')
            
            canvas1 = FigureCanvasTkAgg(fig1, self.fft_linear_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Logarithmic scale FFT (bottom right)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            im2 = ax2.imshow(fft_energy, aspect='auto', vmin=-40, vmax=0)
            ax2.set_title('FFT - Logarithmic Scale')
            ax2.set_xlabel('Frequency Bin')
            ax2.set_ylabel('Time Bin')
            plt.colorbar(im2, ax=ax2, label='dB')
            
            canvas2 = FigureCanvasTkAgg(fig2, self.fft_log_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def clear_results(self):
        """Clear all result displays"""
        for widget in self.scene_frame.winfo_children():
            widget.destroy()
        for widget in self.fft_linear_frame.winfo_children():
            widget.destroy()
        for widget in self.fft_log_frame.winfo_children():
            widget.destroy()


def main():
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
