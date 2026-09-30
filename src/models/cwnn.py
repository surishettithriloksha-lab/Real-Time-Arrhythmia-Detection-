"""Compact real-valued continuous-wavelet neural network."""

import torch
from torch import Tensor, nn
from torch.autograd import Function


class RealMorletCWT(nn.Module):
    """Analytical 32-scale real Morlet transform with normalized convolution kernels."""

    def __init__(self, sample_rate: int = 360, num_scales: int = 32, kernel_size: int = 65) -> None:
        """Initialize Morlet filters spanning 1 to 45 Hz."""
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        f0 = 5.0 / (2.0 * torch.pi)
        frequencies = torch.logspace(
            torch.log10(torch.tensor(1.0)), torch.log10(torch.tensor(45.0)), num_scales
        )
        scales = f0 * sample_rate / frequencies
        time = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        window = torch.hann_window(kernel_size, periodic=False)
        kernels = []
        for scale in scales:
            t = time / scale
            kernel = torch.pi ** (-0.25) * torch.cos(5.0 * t) * torch.exp(-0.5 * t.square())
            kernel = kernel * window
            kernel = kernel / kernel.abs().sum().clamp_min(1e-8)
            kernels.append(kernel)
        self.register_buffer("frequencies", frequencies)
        self.conv = nn.Conv1d(1, num_scales, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv.weight = nn.Parameter(torch.stack(kernels).unsqueeze(1), requires_grad=False)
        self.band_gain = nn.Parameter(torch.ones(num_scales))

    def forward(self, signal: Tensor) -> Tensor:
        """Return absolute CWT energy with shape ``(batch, scales, time)``."""
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)
        if signal.ndim != 3 or signal.size(1) != 1:
            raise ValueError("signal must have shape (batch, 1, time) or (batch, time)")
        return (self.conv(signal) * self.band_gain.view(1, -1, 1)).abs()


class SqueezeExcitation(nn.Module):
    """Lightweight squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        """Initialize channel gates."""
        super().__init__()
        hidden = max(channels // reduction, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply channel-wise attention."""
        weights = self.gate(self.pool(inputs).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return inputs * weights


class SeparableBlock(nn.Module):
    """Depthwise-separable convolution block with batch normalization and SE."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize depthwise and pointwise convolutions."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SqueezeExcitation(out_channels),
            nn.AvgPool2d(2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform one scalogram tensor."""
        return self.block(inputs)


class GradientReversal(Function):
    """Autograd function implementing domain-adversarial gradient reversal."""

    @staticmethod
    def forward(ctx: object, inputs: Tensor, coefficient: float) -> Tensor:
        """Pass features unchanged during the forward pass."""
        ctx.coefficient = coefficient
        return inputs.clone()

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        """Reverse and scale gradients during backpropagation."""
        return -ctx.coefficient * grad_output, None


class CWNN(nn.Module):
    """Compact CWNN classifier with optional 17-class patient discriminator."""

    def __init__(self, num_classes: int = 5, patient_classes: int = 17) -> None:
        """Build the model and its classification/domain heads."""
        super().__init__()
        self.cwt = RealMorletCWT()
        self.backbone = nn.Sequential(
            SeparableBlock(1, 8), SeparableBlock(8, 16), SeparableBlock(16, 24)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(24, num_classes)
        self.patient_discriminator = nn.Sequential(
            nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, patient_classes)
        )

    def features(self, inputs: Tensor) -> Tensor:
        """Extract pooled latent features from raw beats."""
        transformed = self.cwt(inputs).unsqueeze(1)
        return self.pool(self.backbone(transformed)).flatten(1)

    def forward(self, inputs: Tensor, return_domain: bool = False) -> Tensor | dict[str, Tensor]:
        """Classify beats and optionally return DANN patient logits."""
        features = self.features(inputs)
        logits = self.classifier(features)
        if not return_domain:
            return logits
        domain = self.patient_discriminator(GradientReversal.apply(features, 0.1))
        return {"arrhythmia": logits, "patient": domain}
