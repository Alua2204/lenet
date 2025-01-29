# LeNet-5 Fault Injection Testing

This project implements fault injection testing on a quantized LeNet-5 convolutional neural network using PyTorch. The network is trained on the MNIST dataset, quantized to 8-bit integers, and systematically tested by flipping individual bits in its parameters.

## Overview

The testing framework allows both random bit flips across all layers and targeted testing of specific layers and bit positions, measuring how these faults impact the model's classification accuracy. PyTorch was chosen for its ability to modify neural network parameters during runtime testing, as it directly exposes quantized weights through its API.

## Requirements

- Python 3.11
- Dependencies listed in `requirements.txt`

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `lenet.ipynb`: Main Jupyter notebook containing the implementation
- `requirements.txt`: Python package dependencies
- Generated files during execution:
  - `lenet.pt`: Saved trained LeNet model parameters
  - `qlenet.pt`: Saved quantized LeNet model parameters
  - `results.csv`: Random fault injection results
  - `results_c1_5.csv`: Targeted fault injection results for layer c1, bit position 5
  - Various analysis CSV files: `table_4.csv`, `bit_position_impact.csv`, `layer_impact.csv`, `flip_direction_table.csv`
- **Helper Scripts**:
  - `dir.py`: Utilities to organize directories.
  - `chart.py`: Visualization tools for result analysis.

## Usage

#### **Step 1: Train and Quantize the Model**
1. **Open the Notebook**  
   Run the following command to start Jupyter Notebook:
   ```bash
   jupyter notebook lenet.ipynb
   ```
2. **Execute the Notebook**  
   Follow the steps in the notebook to:
   - Train the LeNet-5 model on the MNIST dataset.
   - Quantize the model to 8-bit integers.

#### **Step 2: Perform Fault Injection**
The notebook provides two modes of fault injection:
- **Random Fault Injection**: Flips random bits across all layers and generates results in `results.csv`.
- **Targeted Fault Injection**: Tests specific layers (e.g., `c1`) and specific bit positions (e.g., MSB).

#### **Step 3: Analyze Results**
Results are saved in CSV format for further analysis:
- `bit_position_impact.csv`: Analyzes the impact of specific bit positions.
- `layer_impact.csv`: Evaluates layer-wise sensitivity.
- `flip_direction_table.csv`: Investigates the impact of 0-to-1 vs 1-to-0 flips.

## Fault Injection Details 

#### **Random Fault Injection**
The framework flips bits randomly across all layers and records:
- The impacted layer.
- The bit position.
- Classification errors caused by the fault.

#### **Targeted Fault Injection**
Allows precise testing for:
- Specific layers, e.g., convolutional layers (`c1`, `c3`) or fully connected layers (`f5`, `f7`).
- Specific bit positions, including MSB and LSB.

## Key Features

- Training and quantization of LeNet-5
- Random fault injection across all layers
- Targeted fault injection for specific layers/bits
- Comprehensive result analysis and visualization
- Support for both 0-to-1 and 1-to-0 bit flips

## Results

The framework generates several CSV files containing detailed analysis:
- Layer-wise sensitivity analysis
- Bit position vulnerability analysis
- Flip direction impact analysis
- Targeted testing results


##  Notes and Recommendations 
- Ensure all dependencies are installed before running the notebook.
- Fault injection results can vary depending on the model and test setup.
- Modify `lenet.ipynb` to experiment with new configurations (e.g., testing activations, biases).
- For targeted injections, adjust parameters in the `Fault Injection` section of the notebook.

