"""Dual-channel support-curve adaptation model."""
from __future__ import annotations

import torch
from torch import nn


class DCSA(nn.Module):
    """Predict ZT from composition features, temperature, and support curves.

    The support encoder is permutation invariant over the observed curves.
    """

    def __init__(self, composition_dim: int, hidden: int = 96):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(composition_dim + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.support_point_encoder = nn.Sequential(
            nn.Linear(composition_dim + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.support_curve_encoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, query_x: torch.Tensor, query_t: torch.Tensor,
                support_x: torch.Tensor, support_t: torch.Tensor,
                support_y: torch.Tensor, *, query_base: torch.Tensor | None = None,
                ablation: str = "full") -> torch.Tensor:
        """Return predictions; support tensors have shape ``(batch, shot, dim)``."""
        if query_x.ndim != 2 or support_x.ndim not in (3, 4):
            raise ValueError("query_x must be 2D and support_x must be 3D or 4D")
        if query_base is None:
            query_base = torch.zeros((len(query_x), 1), dtype=query_x.dtype, device=query_x.device)
        q = self.query_encoder(torch.cat([query_x, query_t, query_base], dim=-1))
        s_input = torch.cat([support_x, support_t, support_y], dim=-1)
        if s_input.ndim == 4:
            point = self.support_point_encoder(s_input)
            curve = torch.cat([point.mean(dim=2), point.amax(dim=2)], dim=-1)
            curve = self.support_curve_encoder(curve)
            s = curve.mean(dim=1)
        else:
            # Compatibility with the original summary-based tensor interface.
            s = self.support_curve_encoder(torch.cat([
                self.support_point_encoder(s_input).mean(dim=1),
                self.support_point_encoder(s_input).amax(dim=1)], dim=-1))
        interaction = q * s
        q_input = q
        if ablation == "base_only":
            return query_base.squeeze(-1)
        if ablation == "paper_only":
            q_input = torch.zeros_like(q)
            s_input = s
            interaction_input = torch.zeros_like(q)
        elif ablation == "chemical_only":
            q_input = torch.zeros_like(q)
            s_input = torch.zeros_like(s)
            interaction_input = interaction
        elif ablation == "full":
            q_input, s_input, interaction_input = q, s, interaction
        else:
            raise ValueError(f"unknown ablation: {ablation}")
        delta = self.head(torch.cat([q_input, s_input.expand_as(q), interaction_input], dim=-1)).squeeze(-1)
        return query_base.squeeze(-1) + delta
