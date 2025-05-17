# DisentangleRes
Code for [Disentangling Representations through Multitask Learning](https://openreview.net/forum?id=yVGGtsOgc7&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)) (ICLR 2025).

## Project Description

This repository contains code for training and analyzing neural networks that learn disentangled representations through multitask learning. The project focuses on how neural networks can generalize out-of-distribution and across different tasks by learning to separate task-relevant dimensions in their internal representations.

## Installation

### Requirements

- Python 3.8 or higher
- Git

### Setting up the environment

1. Clone this repository:
   ```bash
   git clone https://github.com/panvaf/DisentangleRes.git
   cd DisentangleRes
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv disentangle
   
   # On Windows
   disentangle\Scripts\activate
   
   # On macOS/Linux
   source disentangle/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install neurogym separately (required for task environments):
   ```bash
   git clone https://github.com/gyyang/neurogym.git
   cd neurogym
   pip install -e .
   cd ..
   ```

## Scripts and Workflow

### Core Scripts

- **train.py**: Trains autoregressive neural networks on specified tasks. Configurable parameters include network architecture, activation functions, noise levels, and training settings.

- **generalize.py**: Evaluates zero-shot, out-of-distribution generalization of trained networks.

- **analyze.py**: Analyzes the representations learned by trained RNNs through dimensionality reduction, fixed point analysis, and other techniques.

- **analyze_transformer.py**: Specific analysis for transformer architectures.

- **sparsity.py**: Analyzes and evaluates sparsity in network representations.

### Supporting Modules

- **tasks.py**: Contains classes for various cognitive tasks used to train the networks.

- **util.py**: Utility functions for analysis, visualization, and data processing.

- **RNN.py**: Implementation of the recurrent neural network architecture.

- **transformer.py**: Implementation of transformer-based architectures for the tasks.

### Typical Workflow

1. **Train networks**: Use `train.py` to train neural networks on various numbers of tasks, for different hyperparameter choices.
   ```bash
   python train.py
   ```

2. **Evaluate generalization**: Use `generalize.py` to test how well trained networks generalize to new tasks. Performs "sweeps" across networks with same hyperparameter choices, and different number of trained tasks.
   ```bash
   python generalize.py
   ```

3. **Analyze representations**: Use `analyze.py` to visualize the representations that the networks have learned.
   ```bash
   python analyze.py
   ```

## Reproducibility

- **Figures/**: Contains code to reproduce all main figures. To do that, please download data from trained networks provided [here](https://gin.g-node.org/pavaf/DisentangleRes), or train and evaluate the networks yourselves by following the instructions in the same link.


## Citation

If you use this code in your research, please cite our paper:

    @inproceedings{
        vafidis2025disentangling,
        title={Disentangling Representations through Multi-task Learning},
        author={Pantelis Vafidis and Aman Bhargava and Antonio Rangel},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=yVGGtsOgc7}
    }