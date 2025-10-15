# Array Processing End-to-End Simulator

**Release v1.0**

## Getting Started

### Prerequisites

#### Install LLVM (Required for DrJit)

DrJit requires LLVM to run on CPU. Install it using conda/mamba:

```bash
conda install -c conda-forge llvm
# or with mamba
mamba install -c conda-forge llvm
```

Set the `DRJIT_LIBLLVM_PATH` environment variable. Run this command to automatically add it to your shell configuration:

```bash
# Detect your shell and add to the appropriate config file
if [ -n "$ZSH_VERSION" ]; then
    echo "export DRJIT_LIBLLVM_PATH=$(find $CONDA_PREFIX/lib -name 'libLLVM*.dylib' 2>/dev/null | head -1)" >> ~/.zshrc
    source ~/.zshrc
elif [ -n "$BASH_VERSION" ]; then
    echo "export DRJIT_LIBLLVM_PATH=$(find $CONDA_PREFIX/lib -name 'libLLVM*.dylib' 2>/dev/null | head -1)" >> ~/.bashrc
    source ~/.bashrc
fi
```

**Verify the setup:**

```bash
echo LLVM PATH: $DRJIT_LIBLLVM_PATH
python -c "import drjit; print('DrJit loaded successfully!')"
```

### Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Generate Data

```bash
python -m e2e.environment.sionna_simple_channel
```

#### 2. Run the Simulator

```bash
python -m e2e.main.main_sionna_blocks
```