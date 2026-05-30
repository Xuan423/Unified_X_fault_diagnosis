import torch
import torch.nn as nn
from einops import rearrange


class SignalProcessingBase(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.device = args.device
        self.to(self.device)

    def forward(self, x):
        raise NotImplementedError

    def test_forward(self):
        test_input = torch.randn(2, self.in_dim, self.in_channels).to(self.device)
        output = self.forward(test_input)
        expected = (2, self.out_dim, self.out_channels)
        assert output.shape == expected, f"input={test_input.shape}, output={output.shape}, expected={expected}"


class SignalProcessingModuleDict(torch.nn.ModuleDict):
    def forward(self, x, key):
        if key not in self:
            raise KeyError(f"No signal processing module found for key: {key}")
        return self[key](x)


class HilbertTransform(SignalProcessingBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = "HT"

    def forward(self, x):
        x = rearrange(x, "b l c -> b c l")
        length = x.shape[-1]
        spectrum = torch.fft.fft(x, dim=2)
        if length % 2 == 0:
            spectrum[..., 1 : length // 2] *= 2
            spectrum[..., length // 2 + 1 :] = 0
        else:
            spectrum[..., 1 : (length + 1) // 2] *= 2
            spectrum[..., (length + 1) // 2 :] = 0
        analytic = torch.fft.ifft(spectrum, dim=2).abs()
        return rearrange(analytic, "b c l -> b l c")


class WaveFilters(SignalProcessingBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = "WF"
        in_channels = args.scale
        self.f_c = nn.Parameter(torch.empty(1, 1, in_channels, device=self.device))
        self.f_b = nn.Parameter(torch.empty(1, 1, in_channels, device=self.device))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.f_c, mean=self.args.f_c_mu, std=self.args.f_c_sigma)
        nn.init.normal_(self.f_b, mean=self.args.f_b_mu, std=self.args.f_b_sigma)

    def filter_generator(self, freq_length):
        omega = torch.linspace(0, 0.5, freq_length, device=self.device).view(1, -1, 1)
        f_b = torch.clamp(self.f_b, min=1e-3).to(self.device)
        f_c = self.f_c.to(self.device)
        self.omega = omega
        return torch.exp(-((omega - f_c) / (2 * f_b)) ** 2)

    def forward(self, x):
        in_dim, in_channels = x.shape[-2], x.shape[-1]
        freq = torch.fft.rfft(x, dim=1, norm="ortho")
        self.filters = self.filter_generator(in_dim // 2 + 1)
        filtered_freq = freq * self.filters[:, :, :in_channels]
        return torch.fft.irfft(filtered_freq, dim=1, norm="ortho").real


class Identity(SignalProcessingBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = "I"

    def forward(self, x):
        return x
