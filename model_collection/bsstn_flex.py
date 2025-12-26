# bsstn_flex.py
# A flexible BS-STN implementation that:
# - Accepts raw time-series input x with shape [B, L, C] (e.g., 64 x 4096 x 4)
# - Builds the functional graph (radius graph by cosine distance on FFT embeddings) and
#   physical graph (simple prior topology) on-the-fly
# - Runs a ChebConv-based STGCN similar to the author code, but with N=C not hard-coded
# - Returns logits (no softmax) so it works with nn.CrossEntropyLoss directly
#
# Requirements: torch_geometric

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import ChebConv, global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import add_self_loops


PhysMode = Literal["complete", "ring", "chain", "identity"]


@dataclass
class BSSTNGraphCfg:
    T: int = 3                 # number of temporal subgraphs
    n_fft: int = 2048          # rFFT size -> Fdim = n_fft//2 + 1 = 1025
    hop: int = 1024            # hop between segments; L=4096 -> segments at 0,1024,2048
    func_eps: float = 0.2      # cosine-distance radius threshold for functional edges
    phys_mode: PhysMode = "complete"  # physical graph prior among sensors per time step
    ensure_connected: bool = True     # if functional graph has no intra-step edges, fallback to complete


class BSSTNFlex(nn.Module):
    """
    Flexible BS-STN-like baseline for multi-sensor time series.

    Input:
        x: Tensor [B, L, C] (float)
    Output:
        logits: Tensor [B, num_classes]  (no softmax)

    Notes:
      - This implementation builds graphs inside forward(), so your Lightning pipeline
        can keep using tensor-level augmentation (AWGN, mixup, etc.) before model(x).
      - Within a single run/batch, C should be constant (otherwise stacking tensors is impossible).
        Across different runs, C can change (2/3/4/...), and this model will adapt dynamically.
    """

    def __init__(
        self,
        num_classes: int,
        max_sensors: int = 8,
        graph_cfg: BSSTNGraphCfg = BSSTNGraphCfg(),
        cheb_k: int = 2,
        h1: int = 256,
        h2: int = 64,
        detach_feature: bool = True,   # treat FFT as fixed embedding (faster; matches "hand-crafted embedding" spirit)
        return_all: bool = False,      # if True: return (logitF, logitD, fusion, logit)
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_sensors = max_sensors
        self.cfg = graph_cfg
        self.cheb_k = cheb_k
        self.h1 = h1
        self.h2 = h2
        self.detach_feature = detach_feature
        self.return_all = return_all

        self.Fdim = self.cfg.n_fft // 2 + 1  # 1025 for n_fft=2048

        # Function graph view
        self.convF1 = ChebConv(self.Fdim, h1, cheb_k)
        self.convF2 = ChebConv(h1, h2, cheb_k)
        self.bnF1 = nn.BatchNorm1d(h1)
        self.bnF2 = nn.BatchNorm1d(h2)
        self.linF = nn.Linear(2 * h2, num_classes)

        # Physical graph view
        self.convD1 = ChebConv(self.Fdim, h1, cheb_k)
        self.convD2 = ChebConv(h1, h2, cheb_k)
        self.bnD1 = nn.BatchNorm1d(h1)
        self.bnD2 = nn.BatchNorm1d(h2)
        self.linD = nn.Linear(2 * h2, num_classes)

        # Fusion head (concat pooled features from both views)
        self.linFusion = nn.Linear(4 * h2, num_classes)

        # Temporal and spatial position embeddings (slice spatial for actual C)
        self.TembF = nn.Parameter(torch.randn(self.cfg.T, self.Fdim))
        self.TembD = nn.Parameter(torch.randn(self.cfg.T, self.Fdim))
        self.SembF = nn.Parameter(torch.randn(max_sensors, self.Fdim))
        self.SembD = nn.Parameter(torch.randn(max_sensors, self.Fdim))

        # Cache physical edge_index by C (nodes per graph = T*C)
        self._phys_edge_cache: dict[int, torch.Tensor] = {}

        # Pre-create window (registered buffer so it moves with .to(device))
        win = torch.hann_window(self.cfg.n_fft)
        self.register_buffer("_fft_window", win, persistent=False)

    # ---------- optional AWGN (you can also keep your existing wgn2 in Lightning) ----------
    @staticmethod
    def add_awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add AWGN to x with the given SNR in dB.
        x: [B, L, C]
        """
        # signal power per (B,C)
        power = x.pow(2).mean(dim=1, keepdim=True)  # [B, 1, C]
        snr = 10.0 ** (snr_db / 10.0)
        noise_power = power / snr
        noise = torch.randn_like(x) * torch.sqrt(noise_power + 1e-12)
        return x + noise

    # ---------- feature extraction ----------
    def _extract_fft_feats(self, x_blc: torch.Tensor) -> torch.Tensor:
        """
        x_blc: [B, L, C]
        Return z: [B, T, C, Fdim]
        """
        B, L, C = x_blc.shape
        T = self.cfg.T
        n_fft = self.cfg.n_fft
        hop = self.cfg.hop
        if L < n_fft + (T - 1) * hop:
            raise ValueError(f"L={L} too short for T={T}, n_fft={n_fft}, hop={hop}")

        # [B, C, L]
        x_bcl = x_blc.permute(0, 2, 1).contiguous()

        # segment starts: 0, hop, 2*hop, ...
        starts = [i * hop for i in range(T)]
        segs = torch.stack([x_bcl[..., s:s + n_fft] for s in starts], dim=2)  # [B, C, T, n_fft]
        segs = segs * self._fft_window.view(1, 1, 1, -1).to(segs.dtype)       # hann window

        spec = torch.fft.rfft(segs, n=n_fft, dim=-1)  # [B, C, T, Fdim] complex
        mag = spec.abs()                               # [B, C, T, Fdim]
        mag = torch.log1p(mag)                         # stabilize

        # per-node normalize over freq dim
        mag = (mag - mag.mean(dim=-1, keepdim=True)) / (mag.std(dim=-1, keepdim=True) + 1e-6)

        z = mag.permute(0, 2, 1, 3).contiguous()       # [B, T, C, Fdim]
        return z

    # ---------- edges ----------
    def _phys_edge_index(self, C: int, device: torch.device) -> torch.Tensor:
        """
        Physical edges are deterministic given C and T: build once and cache on CPU,
        then move to device.
        Nodes are indexed time-major: node_id = t*C + sensor_id.
        """
        if C in self._phys_edge_cache:
            return self._phys_edge_cache[C].to(device)

        T = self.cfg.T
        edges: List[Tuple[int, int]] = []

        def add_undir(u: int, v: int) -> None:
            edges.append((u, v))
            edges.append((v, u))

        # spatial edges within each time step
        for t in range(T):
            base = t * C
            if self.cfg.phys_mode == "identity":
                pass
            elif self.cfg.phys_mode == "chain":
                for i in range(C - 1):
                    add_undir(base + i, base + i + 1)
            elif self.cfg.phys_mode == "ring":
                for i in range(C):
                    add_undir(base + i, base + ((i + 1) % C))
            else:  # complete
                for i in range(C):
                    for j in range(i + 1, C):
                        add_undir(base + i, base + j)

        # temporal edges between adjacent time steps (same sensor)
        for t in range(T - 1):
            for i in range(C):
                add_undir(t * C + i, (t + 1) * C + i)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=T * C)
        self._phys_edge_cache[C] = edge_index.cpu()
        return edge_index.to(device)

    def _func_edge_index_one(self, z_tcf: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Build functional radius graph for ONE sample.
        z_tcf: [T, C, Fdim] (already normalized)
        """
        T, C, Fdim = z_tcf.shape
        eps = float(self.cfg.func_eps)

        edges: List[Tuple[int, int]] = []

        def add_undir(u: int, v: int) -> None:
            edges.append((u, v))
            edges.append((v, u))

        eye = torch.eye(C, device=device, dtype=torch.bool)

        # intra-step edges by cosine distance threshold
        for t in range(T):
            feat = F.normalize(z_tcf[t].to(device), dim=-1)  # [C, F]
            sim = feat @ feat.t()                            # [C, C]
            dist = 1.0 - sim
            mask = (dist <= eps) & (~eye)
            idx = mask.nonzero(as_tuple=False)               # [E, 2]
            base = t * C
            for u, v in idx.tolist():
                add_undir(base + u, base + v)

            if self.cfg.ensure_connected and idx.numel() == 0 and C > 1:
                # fallback: fully connect this time step
                for i in range(C):
                    for j in range(i + 1, C):
                        add_undir(base + i, base + j)

        # temporal edges (always)
        for t in range(T - 1):
            for i in range(C):
                add_undir(t * C + i, (t + 1) * C + i)

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=T * C)
        return edge_index

    def _build_pyg_batches(self, z_btcf: torch.Tensor) -> Tuple[Batch, Batch]:
        """
        z_btcf: [B, T, C, Fdim]
        Return (BatchF, BatchD) for functional and physical views.
        """
        B, T, C, Fdim = z_btcf.shape
        device = z_btcf.device

        # time-major node features: [B, T*C, Fdim]
        x_nodes = z_btcf.reshape(B, T * C, Fdim)

        edgeD = self._phys_edge_index(C, device)

        dataF_list: List[Data] = []
        dataD_list: List[Data] = []
        for b in range(B):
            edgeF = self._func_edge_index_one(z_btcf[b], device)
            xb = x_nodes[b]
            # A is unused in the author forward; we keep a placeholder to avoid attribute errors
            dataF_list.append(Data(x=xb, edge_index=edgeF, A=torch.tensor(0, device=device)))
            dataD_list.append(Data(x=xb, edge_index=edgeD, A=torch.tensor(0, device=device)))

        return Batch.from_data_list(dataF_list), Batch.from_data_list(dataD_list)

    # ---------- core forward ----------
    def _mid_step_idx(self, B: int, C: int, device: torch.device) -> torch.Tensor:
        """
        Indices for nodes at the middle time step (t = T//2) for each graph in the batch.
        Assumes fixed nodes per graph = T*C and Batch node order is graph-major.
        """
        T = self.cfg.T
        mid_t = T // 2
        stride = T * C
        start = mid_t * C
        base = (torch.arange(B, device=device) * stride).unsqueeze(1)               # [B,1]
        offs = torch.arange(start, start + C, device=device).unsqueeze(0)           # [1,C]
        idx = (base + offs).reshape(-1)                                             # [B*C]
        return idx

    def forward(
        self,
        x: torch.Tensor,
        snr_db: Optional[float] = None,   # optional internal AWGN
    ):
        """
        x: [B, L, C]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B,L,C], got {tuple(x.shape)}")
        x = x.float()
        B, L, C = x.shape
        if C > self.max_sensors:
            raise ValueError(f"C={C} exceeds max_sensors={self.max_sensors}. Increase max_sensors.")

        if snr_db is not None:
            x = self.add_awgn(x, float(snr_db))

        # FFT embeddings
        if self.detach_feature:
            with torch.no_grad():
                z = self._extract_fft_feats(x)  # [B,T,C,F]
        else:
            z = self._extract_fft_feats(x)

        # Build PyG batches
        dataF, dataD = self._build_pyg_batches(z)

        # Add temporal/spatial embeddings (slice spatial embeddings to actual C)
        T = self.cfg.T
        SembF = self.SembF[:C].to(z.device)
        SembD = self.SembD[:C].to(z.device)

        # data?.x is graph-major: [B*T*C, F]
        xF = dataF.x.view(B, T, C, self.Fdim).permute(0, 2, 1, 3)  # [B,C,T,F]
        xF = xF + self.TembF.to(z.device).unsqueeze(0).unsqueeze(1) + SembF.unsqueeze(0).unsqueeze(2)
        xF = xF.permute(0, 2, 1, 3).reshape(B * T * C, self.Fdim)

        xD = dataD.x.view(B, T, C, self.Fdim).permute(0, 2, 1, 3)  # [B,C,T,F]
        xD = xD + self.TembD.to(z.device).unsqueeze(0).unsqueeze(1) + SembD.unsqueeze(0).unsqueeze(2)
        xD = xD.permute(0, 2, 1, 3).reshape(B * T * C, self.Fdim)

        # Replace batch x with embedded x
        dataF.x = xF
        dataD.x = xD

        # Mid-step node indices (for pooling)
        idx = self._mid_step_idx(B, C, z.device)
        bF = dataF.batch[idx]
        bD = dataD.batch[idx]

        # Function view
        hF = F.relu(self.bnF1(self.convF1(dataF.x, dataF.edge_index)))
        hF = F.relu(self.bnF2(self.convF2(hF, dataF.edge_index)))
        featF = torch.cat([gmp(hF[idx], bF), gap(hF[idx], bF)], dim=1)  # [B, 2*h2]
        logitF = self.linF(featF)

        # Physical view
        hD = F.relu(self.bnD1(self.convD1(dataD.x, dataD.edge_index)))
        hD = F.relu(self.bnD2(self.convD2(hD, dataD.edge_index)))
        featD = torch.cat([gmp(hD[idx], bD), gap(hD[idx], bD)], dim=1)
        logitD = self.linD(featD)

        fusion = torch.cat([featF, featD], dim=1)  # [B, 4*h2]
        logit = self.linFusion(fusion)

        if self.return_all:
            return logitF, logitD, fusion, logit
        return logit

if __name__ == "__main__":
    # simple test
    model = BSSTNFlex(num_classes=5, max_sensors=4)
    x = torch.randn(2, 4096, 4)  # [B,L,C]
    logits = model(x)
    print(logits.shape)  # should be [2,5]