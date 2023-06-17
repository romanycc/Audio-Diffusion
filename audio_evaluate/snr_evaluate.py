import torch
from torchmetrics import SignalNoiseRatio

# Generate some dummy audio signals
signal = torch.tensor([0.5, 0.8, 1.0, 0.7])
noise = torch.tensor([0.1, 0.2, 0.3, 0.1])
audio = signal + noise

# Create an instance of SignalNoiseRatio metric
snr_metric = SignalNoiseRatio(zero_mean=False)

# Calculate SNR
snr = snr_metric(audio, signal)

# Print the SNR value
print("SNR:", snr)
