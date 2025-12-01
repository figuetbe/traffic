#!/usr/bin/env python3
"""
Conditional Flow Matching (CFM) prediction for aircraft trajectories.

This module implements the CFM prediction method for the Traffic library,
providing generative traj prediction using a conditional flow matching model.
"""

import json
import math
from datetime import timedelta
from typing import Optional

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from ...core.flight import Flight
from ...core.traffic import Traffic
from . import PredictorBase

# ---------------------- Model Architecture ----------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TimeEmbedding(nn.Module):
    """Time step embedding for flow matching."""

    def __init__(self, d_model: int, hidden: int = 256, emb_dim: int = 128):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.linspace(
                    0, math.log(10_000), emb_dim // 2, dtype=torch.float32
                )
            ),
        )
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, d_model)
        )

    def forward(self, t):
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        ang = t.unsqueeze(-1) * self.freqs  # (B, F)
        temb = torch.cat(
            [torch.sin(ang), torch.cos(ang)], dim=-1
        )  # (B, emb_dim)
        return self.proj(temb)  # (B, d_model)


class HistoryEncoder(nn.Module):
    """Transformer encoder for processing flight history sequences."""

    def __init__(
        self, in_dim, d_model, nhead, num_layers, ff, dropout, context_dim=3
    ):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        self.context_proj = (
            nn.Linear(context_dim, d_model) if context_dim > 0 else None
        )
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=1024)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context=None):
        z = self.input(x)
        z = self.pos(z)
        if self.context_proj is not None and context is not None:
            c = self.context_proj(context).unsqueeze(1)  # (B,1,D)
            z = torch.cat([c, z], dim=1)
        return self.norm(self.enc(z))


class FutureDenoiser(nn.Module):
    """Transformer decoder for denoising future trajectory predictions."""

    def __init__(self, in_dim, d_model, nhead, num_layers, ff, dropout):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=512)
        self.t_proj_tokens = nn.Linear(d_model, d_model)
        self.t_proj_memory = nn.Linear(d_model, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, in_dim)

    def forward(self, xt, mem, t_emb):
        z = self.pos(self.input(xt))
        z = z + self.t_proj_tokens(t_emb).unsqueeze(1)
        mem = mem + self.t_proj_memory(t_emb).unsqueeze(1)
        return self.output(self.norm(self.dec(tgt=z, memory=mem)))


class FlowMatchingModel(nn.Module):
    """Flow Matching Model for conditional generative trajectory prediction.

    This model combines historical flight data with contextual information
    to generate probabilistic predictions of future aircraft trajectories.

    Args:
        d_model: Model dimension for transformer layers
        nhead: Number of attention heads
        enc_layers: Number of encoder layers
        dec_layers: Number of decoder layers
        ff: Feed-forward network dimension
        dropout: Dropout probability
        in_dim: Input feature dimension
        context_dim: Context feature dimension
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        enc_layers=6,
        dec_layers=8,
        ff=4 * 512,
        dropout=0.1,
        in_dim=7,
        context_dim=8,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.encoder = HistoryEncoder(
            in_dim,
            d_model,
            nhead,
            enc_layers,
            ff,
            dropout,
            context_dim=context_dim,
        )
        self.time_emb = TimeEmbedding(d_model=d_model)
        self.denoiser = FutureDenoiser(
            in_dim, d_model, nhead, dec_layers, ff, dropout
        )

    def forward(self, x_hist, x_t, t_scalar, context):
        mem = self.encoder(x_hist, context)
        t_emb = self.time_emb(t_scalar)
        return self.denoiser(x_t, mem, t_emb)


def load_model_checkpoint(
    checkpoint_path: str, device=None
) -> FlowMatchingModel:
    """Load model checkpoint and return configured model.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on (auto-detects if None)

    Returns:
        Loaded FlowMatchingModel instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle potential module prefix issues
    state = {
        k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()
    }

    # Get model configuration from checkpoint or use defaults
    cfg = ckpt.get(
        "model_cfg",
        dict(
            d_model=512,
            nhead=8,
            enc_layers=6,
            dec_layers=8,
            ff=4 * 512,
            dropout=0.1,
            in_dim=7,
            context_dim=8,
        ),
    )

    # Initialize and load model
    model = FlowMatchingModel(**cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


# ---------------------- Inference Utilities ----------------------


@torch.no_grad()
def sample_future_heun(
    model: torch.nn.Module,
    x_hist: torch.Tensor,
    context: torch.Tensor,
    T_out: int,
    n_steps: int = 64,
    G: float = 1.0,
    use_autocast: bool = True,
) -> torch.Tensor:
    """
    Heun integrator for Conditional Flow Matching, returning normalized futures.

    Args:
        model: torch.nn.Module with forward (x_hist, x_t, t_scalar, context)
        x_hist: (B, L, D) normalized, aircraft-centric history
        context: (B, C) normalized context
        T_out: number of future steps
        n_steps: time discretization of the flow (default 64)
        G: guidance/scaling factor for the velocity field
        use_autocast: enable CUDA autocast if available

    Returns:
        x: (B, T_out, D) normalized future sequence
    """
    model.eval()
    B, _, D = x_hist.shape

    # Init the future traj with random noise from standard normal distribution
    x = torch.randn(B, T_out, D, device=x_hist.device, dtype=x_hist.dtype)

    # Calculate time step size for discretizing the continuous flow
    dt = 1.0 / n_steps

    amp_enabled = (
        (x_hist.device.type == "cuda") or (x_hist.device.type == "mps")
    ) and use_autocast
    amp_dtype = torch.bfloat16

    # Perform Heun integration over n_steps to solve the flow ODE
    for k in range(n_steps):
        # Calculate current and next time points for this integration step
        # Clamp to avoid numerical issues at t=1.0
        t0v = min(k * dt, 1.0 - 1e-6)
        t1v = min((k + 1) * dt, 1.0 - 1e-6)

        # Create time tensors for the entire batch
        t0 = torch.full((B, 1), t0v, device=x_hist.device, dtype=x_hist.dtype)
        t1 = torch.full((B, 1), t1v, device=x_hist.device, dtype=x_hist.dtype)

        with torch.amp.autocast(
            device_type=x_hist.device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            # Compute velocity field at current time t0 and position x
            v1 = model(x_hist, x, t0, context)

            # Take Euler step to estimate position at next time (predictor step)
            x_pred = x + (G * v1) * dt

            # Compute velocity field at predicted position and next time t1
            v2 = model(x_hist, x_pred, t1, context)

            # Take weighted avg of velocities and update pos (corrector step)
            x = x + 0.5 * (G * v1 + G * v2) * dt
    return x


@torch.no_grad()
def denorm_seq_to_global(
    seq_norm: torch.Tensor,
    ctx_norm: torch.Tensor,
    feat_mean: torch.Tensor | np.ndarray,
    feat_std: torch.Tensor | np.ndarray,
    ctx_mean: torch.Tensor | np.ndarray,
    ctx_std: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """
    Convert normalized, aircraft-centric sequence back to global frame.

    Args:
        seq_norm: (B, T, D) normalized sequence
        ctx_norm: (B, C) normalized context
        feat_mean/std: per-feature stats of length D
        ctx_mean/std: per-context stats of length >= 5 (x0,y0,z0,cos,sin,...)

    Returns:
        (B, T, D) in global coordinates/units
    """
    # Extract dimensions for tensor operations
    B, T, D = seq_norm.shape

    # Convert feature statistics to tensors with proper shape for broadcasting
    # Reshape to (1, 1, D) so they can be broadcasted across batch and time dim
    fm = torch.as_tensor(
        feat_mean, dtype=seq_norm.dtype, device=seq_norm.device
    ).view(1, 1, -1)
    fs = torch.as_tensor(
        feat_std, dtype=seq_norm.dtype, device=seq_norm.device
    ).view(1, 1, -1)

    # Denorma the sequence using feature stats: seq = seq_norm * std + mean
    seq = seq_norm * fs + fm

    # Get context dimension and convert context statistics to tensors
    C = ctx_norm.size(-1)
    cm = torch.as_tensor(
        ctx_mean[:C], dtype=ctx_norm.dtype, device=ctx_norm.device
    ).view(1, C)
    cs = torch.as_tensor(
        ctx_std[:C], dtype=ctx_norm.dtype, device=ctx_norm.device
    ).view(1, C)

    # Denormalize the context: ctx_raw = ctx_norm * std + mean
    ctx_raw = ctx_norm * cs + cm

    # Extract cosine and sine values for rotation matrix (aircraft heading)
    # These represent the aircraft's orientation in the global frame
    c = ctx_raw[:, 3:4]  # cos(heading)
    s = ctx_raw[:, 4:5]  # sin(heading)

    # Rotate position coordinates from aircraft-centric to global frame
    # Use R^T (transpose/inv of rotation matrix): [cos θ, sin θ; -sin θ, cos θ]
    x_local = seq[..., 0]  # aircraft-centric x (forward direction)
    y_local = seq[..., 1]  # aircraft-centric y (right direction)
    x_global = c * x_local + s * y_local  # compute before assignment
    y_global = -s * x_local + c * y_local
    seq[..., 0] = x_global
    seq[..., 1] = y_global

    # Rotate velocity coordinates using the same rotation matrix
    # Velocities transform the same way as positions under rotation
    vx_local = seq[..., 3]  # aircraft-centric velocity x
    vy_local = seq[..., 4]  # aircraft-centric velocity y
    vx_global = c * vx_local + s * vy_local
    vy_global = -s * vx_local + c * vy_local
    seq[..., 3] = vx_global
    seq[..., 4] = vy_global

    # Add reference position to convert from relative to absolute coordinates
    # ctx_raw[:, :3] contains the aircraft's reference position (x0, y0, z0)
    ref_xyz = ctx_raw[:, :3].view(B, 1, 3)
    seq[..., :3] = seq[..., :3] + ref_xyz

    # Return the sequence in global coordinates with proper units
    return seq


def rotate_xy_inplace(arr: np.ndarray, c: np.ndarray, s: np.ndarray) -> None:
    """Rotate x,y coordinates in-place using rotation matrix [c, -s; s, c]."""
    x = arr[..., 0].astype(np.float64)
    y = arr[..., 1].astype(np.float64)
    arr[..., 0] = c[:, None] * x - s[:, None] * y
    arr[..., 1] = s[:, None] * x + c[:, None] * y
    vx = arr[..., 3].astype(np.float64)
    vy = arr[..., 4].astype(np.float64)
    arr[..., 3] = c[:, None] * vx - s[:, None] * vy
    arr[..., 4] = s[:, None] * vx + c[:, None] * vy


def aircraft_centric_transform(
    X_raw: np.ndarray, Y_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform global coordinates to aircraft-centric frame."""
    refs = X_raw[:, -1, :3].copy()
    X_t = X_raw.copy()
    Y_t = Y_raw.copy()
    X_t[..., :3] -= refs[:, None, :]
    Y_t[..., :3] -= refs[:, None, :]

    vx_last = X_raw[:, -1, 3].astype(np.float64)
    vy_last = X_raw[:, -1, 4].astype(np.float64)
    vz_last = X_raw[:, -1, 5].astype(np.float64)
    gs_last = np.hypot(vx_last, vy_last).astype(np.float64)
    eps = 1e-8
    c = np.where(gs_last > eps, vy_last / (gs_last + eps), 1.0)
    s = np.where(gs_last > eps, vx_last / (gs_last + eps), 0.0)

    rotate_xy_inplace(X_t, c, s)
    rotate_xy_inplace(Y_t, c, s)

    psi_rate_last = X_raw[:, -1, 6].astype(np.float32)
    C_raw = np.stack(
        [
            refs[:, 0].astype(np.float32),
            refs[:, 1].astype(np.float32),
            refs[:, 2].astype(np.float32),
            c.astype(np.float32),
            s.astype(np.float32),
            gs_last.astype(np.float32),
            vz_last.astype(np.float32),
            psi_rate_last,
        ],
        axis=1,
    )
    return X_t, Y_t, C_raw


# ---------------------- CFM Predictor Class ----------------------


class CFMPredict(PredictorBase):
    """Conditional Flow Matching predictor for flight trajectories.

    This class implements trajectory prediction using a trained conditional
    flow matching model. It takes a Flight object with at least 60 seconds
    of historical data and generates probabilistic predictions of future
    trajectory segments.
    """

    method_name = "cfm"

    def __init__(
        self,
        model_path: str,
        stats_path: str,
        device: Optional[str] = None,
        n_samples: int = 10,
        n_steps: int = 64,
        guidance_scale: float = 1.0,
    ):
        """Initialize the CFM predictor.

        Args:
            model_path: Path to the trained model checkpoint
            stats_path: Path to the normalization statistics JSON file
            device: Device to run inference on ('cuda', 'cpu', etc.)
            n_samples: Number of trajectory samples to generate
            n_steps: Number of integration steps for flow ODE
            guidance_scale: Scale factor for velocity field guidance
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model and stats
        self.model = load_model_checkpoint(model_path, self.device)
        with open(stats_path, "r") as f:
            self.norm_stats = json.load(f)

        self.n_samples = n_samples
        self.n_steps = n_steps
        self.guidance_scale = guidance_scale

        # Extract stats
        self.feat_mean = np.array(
            self.norm_stats["feat_mean"], dtype=np.float32
        )
        self.feat_std = np.array(self.norm_stats["feat_std"], dtype=np.float32)
        self.ctx_mean = np.array(self.norm_stats["ctx_mean"], dtype=np.float32)
        self.ctx_std = np.array(self.norm_stats["ctx_std"], dtype=np.float32)

    def _resample_predictions(
        self, predictions: torch.Tensor, T_out: int, sampling_rate: float
    ) -> torch.Tensor:
        """
        Resample predictions from 5s intervals to desired sampling rate.

        predictions: (n_samples, 12, D) at +5,+10,…,+60 seconds
        sampling_rate: new step in seconds.
        We resample onto [5, 5+sampling_rate, 5+2*sampling_rate, …, 60].
        """
        assert predictions.dim() == 3, "predictions must be (S, T, D)"

        S, T, D = predictions.shape

        # Original prediction times are +5s … +60s (12 points)
        original_times = np.arange(5, 61, 5, dtype=float)  # [5,10,...,60]

        new_times = np.arange(
            5, 60 + 1e-6, sampling_rate, dtype=float
        )  # [5, 5+sampling_rate, 5+2*sampling_rate, ..., ~60]

        # Interpolate along time for each feature independently
        out = torch.empty(
            (S, len(new_times), D),
            dtype=predictions.dtype,
            device=predictions.device,
        )

        pred_np = predictions.detach().cpu().numpy()  # (S,T,D)

        for f in range(D):
            vals = pred_np[:, :, f]  # (S,T)
            res = np.vstack(
                [
                    np.interp(new_times, original_times, vals_i)
                    for vals_i in vals
                ]
            )  # (S, len(new_times))
            out[:, :, f] = torch.from_numpy(res).to(out.device, out.dtype)

        return out

    def preprocess_flight(
        self, flight: Flight
    ) -> tuple[np.ndarray, pd.Timestamp]:
        """Preprocess flight data for model input.

        Args:
            flight: Flight object with trajectory data

        Returns:
            Tuple of (processed_features, last_timestamp)
        """
        # Get last 60 seconds of data
        window = flight.last(seconds=60)
        if window is None or len(window.data) < 60:
            raise ValueError("Flight must have at least 60 seconds of data")

        df = window.data.copy()

        # Ensure proper column names and types
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Assume input is already sorted and resampled to 60 samples at 1Hz
        if len(df) != 60:
            raise ValueError(f"Expected exactly 60 samples, got {len(df)}")

        # Get last timestamp for prediction start
        last_timestamp = df["timestamp"].iloc[-1]

        # Convert coordinates to LV95
        import pyproj

        crs_lv95 = pyproj.CRS.from_epsg(2056)
        crs_wgs84 = pyproj.CRS.from_epsg(4326)
        to_lv95 = pyproj.Transformer.from_crs(
            crs_wgs84, crs_lv95, always_xy=True
        )

        x_coords, y_coords = to_lv95.transform(
            df["longitude"].to_numpy(), df["latitude"].to_numpy()
        )
        df["x"] = x_coords
        df["y"] = y_coords

        # Convert units
        KNOTS2MPS = 0.5144444444444444
        FTPM2MPS = 0.3048 / 60.0
        track_rad = np.deg2rad(df["track"].to_numpy())
        spd_mps = df["groundspeed"].to_numpy() * KNOTS2MPS
        df["vx"] = spd_mps * np.sin(track_rad)
        df["vy"] = spd_mps * np.cos(track_rad)
        df["z"] = df["altitude"] * 0.3048
        df["vz"] = df["vertical_rate"] * FTPM2MPS

        # Compute psi_rate exactly like training (group shift over one series)
        vx = df["vx"].to_numpy()
        vy = df["vy"].to_numpy()
        vxm = np.roll(vx, 1)
        vxm[0] = vx[0]
        vym = np.roll(vy, 1)
        vym[0] = vy[0]
        cross = vxm * vy - vym * vx
        dot = vxm * vx + vym * vy
        psi_rate = -np.arctan2(cross, dot)
        df["psi_rate"] = pd.Series(psi_rate, index=df.index)
        df["psi_rate"] = np.nan_to_num(
            df["psi_rate"], nan=0.0, posinf=0.0, neginf=0.0
        )
        df["psi_rate"] = np.clip(df["psi_rate"], -0.25, 0.25)

        # Extract features
        features = ["x", "y", "z", "vx", "vy", "vz", "psi_rate"]
        X_raw = df[features].to_numpy().astype(np.float32)

        return X_raw, last_timestamp

    def predict(self, flight: Flight) -> Flight | Traffic:
        """Generate trajectory predictions for a flight.

        Args:
            flight: Flight object with historical trajectory data

        Returns:
            Flight object containing the predicted trajectory if n_samples=1,
            or Traffic object containing multiple predicted trajs if n_samples>1
        """
        # Preprocess flight data
        X_raw, last_timestamp = self.preprocess_flight(flight)

        # Apply aircraft-centric transform
        X_raw_b = X_raw[None, :, :]
        Y_dummy = np.zeros((1, 1, X_raw.shape[1]), dtype=X_raw.dtype)
        X_t_b, _, C_raw_b = aircraft_centric_transform(X_raw_b, Y_dummy)
        X_t, C_raw = X_t_b[0], C_raw_b[0]

        # Normalize
        X_norm = ((X_t - self.feat_mean) / self.feat_std).astype(np.float32)
        C_norm = (
            (C_raw - self.ctx_mean[: len(C_raw)]) / self.ctx_std[: len(C_raw)]
        ).astype(np.float32)

        # Convert to tensors
        x_hist = torch.from_numpy(X_norm).unsqueeze(0).to(self.device)
        ctx = torch.from_numpy(C_norm).unsqueeze(0).to(self.device)

        # Generate multiple samples by repeating inputs
        repeated_hist = x_hist.repeat(self.n_samples, 1, 1)
        repeated_ctx = ctx.repeat(self.n_samples, 1)

        # Sample futures at the model's native stride (5s → 12 steps up to +60s)
        T_out = 12
        futures_norm = sample_future_heun(
            self.model,
            repeated_hist,
            repeated_ctx,
            T_out=T_out,
            n_steps=self.n_steps,
            G=self.guidance_scale,
        )

        # Denormalize and convert back to global coordinates
        ctx_rep = ctx.repeat(self.n_samples, 1)
        futures_global = denorm_seq_to_global(
            futures_norm,
            ctx_rep,
            self.feat_mean,
            self.feat_std,
            self.ctx_mean,
            self.ctx_std,
        )

        # Resample preds from 5s intervals to 1s intervals for smoother output
        futures_global = self._resample_predictions(
            futures_global, T_out, sampling_rate=1
        )

        # Convert to lat/lon/alt
        futures_np = futures_global.detach().cpu().numpy()
        import pyproj

        crs_lv95 = pyproj.CRS.from_epsg(2056)
        crs_wgs84 = pyproj.CRS.from_epsg(4326)
        to_wgs84 = pyproj.Transformer.from_crs(
            crs_lv95, crs_wgs84, always_xy=True
        )

        predicted_flights = []
        for i in range(self.n_samples):
            lon, lat = to_wgs84.transform(
                futures_np[i, :, 0], futures_np[i, :, 1]
            )
            alt_ft = futures_np[i, :, 2] / 0.3048

            # Create tstamps for preds (1s intervals from last_timestamp + 1s)
            pred_timestamps = [
                last_timestamp + timedelta(seconds=j + 1)
                for j in range(60)  # 60 seconds at 1Hz
            ][
                : len(lon)
            ]  # Ensure we don't have more timestamps than predictions

            # Create flight data for this prediction
            flight_id = (
                flight.identifier
                if hasattr(flight, "identifier")
                else "predicted"
            )
            if self.n_samples > 1:
                # Add sample index to distinguish multiple predictions
                flight_id = f"{flight_id}_sample_{i}"

            pred_data = pd.DataFrame(
                {
                    "timestamp": pred_timestamps,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt_ft,
                    "flight_id": flight_id,
                }
            )

            # Create flight with only predicted data (no historical data)
            predicted_flights.append(Flight(pred_data))

        # Return single Flight if n_samples=1, otherwise return Traffic
        if self.n_samples == 1:
            return predicted_flights[0]
        else:
            return Traffic.from_flights(predicted_flights)
