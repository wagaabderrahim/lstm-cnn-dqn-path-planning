# Enhancing CNN-DQN with LSTM-Based Temporal Modeling for Mobile Robot Path Planning

This repository contains the official implementation of the paper **"Enhancing CNN-DQN with LSTM-Based Temporal Modeling for Mobile Robot Path Planning"**.

The proposed **LSTM-CNN-DQN** architecture integrates Long Short-Term Memory (LSTM) units into a Convolutional Neural Network Deep Q-Network (CNN-DQN) framework. This enables the agent to learn from sequential state transitions and make decisions grounded in historical context, solving the fundamental lack of temporal memory in standard CNN-DQN methods.

## Features
- **GridWorld Environment**: A customizable 2D grid environment ($15 \times 15$) with random static or dynamic obstacles.
- **Local Observation Window**: Uses an egocentric $11 \times 11$ sliding window to reduce computational complexity and enable generalization.
- **Architectures Included**:
  - Baseline CNN-DQN (Global & Local)
  - Dueling CNN-DQN
  - Proposed LSTM-CNN-DQN
  - Frame-Stack CNN-DQN Baseline
- **Advanced RL Techniques**: Prioritized Experience Replay (PER), Double DQN (DDQN), and Dueling networks.
- **Evaluation Scripts**: Comprehensive evaluation for static grids, ablation studies, and dynamic environments (moving obstacles).

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/wagaabderrahim/lstm-cnn-dqn-robot-navigation.git
cd lstm-cnn-dqn-robot-navigation
pip install -r requirements.txt
```

## Repository Structure
- `cnn_dqn.py`: Core logic including the `GridWorld` environment, Replay Buffers, CNN-DQN, and LSTM-CNN-DQN architectures.
- `complete_evaluation_and_plots.py`: Scripts for evaluating trained models on 100 unseen test environments and generating performance plots.
- `evaluate_dynamic_environments.py`: Evaluation script for non-stationary environments where obstacles move periodically.
- `train_framestack.py`: Script to train and evaluate the Frame-Stack CNN-DQN baseline.
- `ablation_study.py`: Ablation study configurations and evaluations.

## Data Availability
The synthetic environments and raw logs used in this study can be reproduced by running the environment generation modules. 
For convenience, we also provide:
- `test_environments.npz`: A compressed NumPy archive containing the exact 100 test environments (15x15 grids) used for evaluation.
- `env_images_compressed.zip`: Visualizations (PNG format) of all 100 test environments.

Pre-trained weights for the baseline and LSTM models will be made available in the `models/` directory.

## Citation
If you find this code useful in your research, please cite our paper:
```bibtex
@article{waga2026enhancing,
  title={Enhancing CNN-DQN with LSTM-Based Temporal Modeling for Mobile Robot Path Planning},
  author={Waga, Abderrahim and Benhlima, Said and Bekri, Ali and Saber, Fatima Zahrae and Abdouni, Jawad and Mzili, Toufik and Regragui, Ahmed},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026}
}
```
