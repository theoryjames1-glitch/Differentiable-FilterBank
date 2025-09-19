Awesome 🚀 Let’s build a **Differentiable FilterBank** in PyTorch — basically a *neural EQ layer* that can be dropped into a DDSP/FM6 system.

This gives us **multiple parallel differentiable IIR filters** (Butterworth, Chebyshev, etc.) initialized from classical DSP, but fully trainable with autograd. Perfect for analysis-synthesis, spectral shaping, or adaptive timbre control.

---

# 🎛 Differentiable FilterBank

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal

# -----------------------------------
# Utility: design filter & convert to SOS
# -----------------------------------
def design_filter(ftype, order, cutoff, fs=2.0, btype="low"):
    """
    ftype: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
    order: filter order
    cutoff: cutoff (normalized 0..1 if fs=2.0)
    fs: sample rate
    btype: 'low', 'high', 'bandpass', 'bandstop'
    """
    if ftype == "butter":
        z, p, k = signal.butter(order, cutoff, btype=btype, fs=fs, output="zpk")
    elif ftype == "cheby1":
        z, p, k = signal.cheby1(order, 1, cutoff, btype=btype, fs=fs, output="zpk")
    elif ftype == "cheby2":
        z, p, k = signal.cheby2(order, 20, cutoff, btype=btype, fs=fs, output="zpk")
    elif ftype == "ellip":
        z, p, k = signal.ellip(order, 1, 40, cutoff, btype=btype, fs=fs, output="zpk")
    elif ftype == "bessel":
        z, p, k = signal.bessel(order, cutoff, btype=btype, fs=fs, output="zpk", norm="phase")
    else:
        raise ValueError("Unsupported filter type")
    sos = signal.zpk2sos(z, p, k, fs=fs)
    return sos

# -----------------------------------
# Differentiable Biquad Section
# -----------------------------------
class DifferentiableBiquad(nn.Module):
    def __init__(self, b, a):
        super().__init__()
        b = b / a[0]
        a = a / a[0]
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(a[1:], dtype=torch.float32))
        self.register_buffer("zi", torch.zeros(2))

    def forward(self, x):
        y = []
        z1, z2 = self.zi
        for n in range(x.shape[-1]):
            xn = x[..., n]
            yn = self.b[0]*xn + z1
            z1_new = self.b[1]*xn - self.a[0]*yn + z2
            z2_new = self.b[2]*xn - self.a[1]*yn
            z1, z2 = z1_new, z2_new
            y.append(yn)
        self.zi = torch.stack([z1.detach(), z2.detach()])
        return torch.stack(y, dim=-1)

# -----------------------------------
# Differentiable Filter (IIR from SOS)
# -----------------------------------
class DifferentiableIIR(nn.Module):
    def __init__(self, sos):
        super().__init__()
        self.sections = nn.ModuleList([
            DifferentiableBiquad(s[:3], s[3:]) for s in sos
        ])

    def forward(self, x):
        y = x
        for sec in self.sections:
            y = sec(y)
        return y

# -----------------------------------
# Differentiable FilterBank (multi-band EQ)
# -----------------------------------
class DifferentiableFilterBank(nn.Module):
    def __init__(self, bands, fs=2.0):
        """
        bands: list of dicts [{ "ftype":"butter", "order":2, "cutoff":0.2, "btype":"low" }, ...]
        """
        super().__init__()
        self.filters = nn.ModuleList()
        for band in bands:
            sos = design_filter(band["ftype"], band["order"], band["cutoff"], fs=fs, btype=band["btype"])
            self.filters.append(DifferentiableIIR(sos))

    def forward(self, x):
        # Parallel filtering, then stack outputs
        outs = [filt(x) for filt in self.filters]
        return torch.stack(outs, dim=1)  # shape: [batch, bands, time]
```

---

# ⚡ Example: Trainable 3-Band EQ

```python
# Design a 3-band EQ: low, mid, high
bands = [
    {"ftype":"butter", "order":2, "cutoff":0.2, "btype":"low"},
    {"ftype":"butter", "order":2, "cutoff":[0.2,0.6], "btype":"bandpass"},
    {"ftype":"butter", "order":2, "cutoff":0.6, "btype":"high"},
]

fbank = DifferentiableFilterBank(bands, fs=2.0)

# Input: white noise
x = torch.randn(1, 2048)
y = fbank(x)  # shape [1, 3, 2048]

print("FilterBank output:", y.shape)
```

---

# 🧪 Training Idea

You can train this **FilterBank** as a **learnable spectral front-end**:

```python
optimizer = torch.optim.Adam(fbank.parameters(), lr=1e-3)

for step in range(50):
    x = torch.randn(1, 1024)
    y = fbank(x)  # [1, bands, T]

    # Example loss: encourage smooth energy distribution
    energy = (y**2).mean(-1)  # [batch, bands]
    target = torch.tensor([[1.0, 0.5, 0.2]])  # desired energy profile
    loss = F.mse_loss(energy, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | Loss={loss.item():.6f}")
```

---

✅ This gives you a **fully differentiable filterbank** that:

* Starts as a known DSP design (Butterworth, Chebyshev, etc.).
* Learns **band edges, gains, Q** via gradient descent.
* Can act as a **spectral encoder** for DDSP/FM6 or as an **adaptive EQ**.

---

Would you like me to **extend this further into a Differentiable Parametric EQ** (bands with trainable center frequency, gain, Q-factor)? That would give you a direct bridge to how mixing consoles & synthesizers handle sound shaping.
