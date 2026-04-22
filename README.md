# Exp-Matrix-PINN: Matrix Exponential Physic Informed Neural Networks

## 🛠️ Installation

### Prerequisites
- Python 3.12.11
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/exp-matrix-pinn.git
cd exp-matrix-pinn

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12.11
source .venv/bin/activate

# Install dependencies via uv
uv sync --all-groups

# Install pre-commit
pre-commit install
```
