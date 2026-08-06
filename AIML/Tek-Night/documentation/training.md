# Enviroment

- Framework: PyTorch

- GPU used: RTX 4070 (8 GB VRAM)

- CPU used: i9 14900HX

- CUDA: True

# Hyperparameters

- Optimizer: Adam

- Learning rate: 0.0001

- Loss: L1 loss

- Epochs: 50

- Batch size: 16

# Pipeline

Load batch
    |
Move to device
    |
Forward pass
    |
Calculate L1 loss
    |
Backprop
    |
Update weights