#!/usr/bin/env python3
"""
Bayesian Network (BN) prediction for aircraft trajectories.

This module implements the BN prediction method for the Traffic library,
providing generative trajectory prediction using an autoregressive Gaussian
Bayesian Network trained on residuals.
"""

from typing import Optional
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import timedelta

from ...core.flight import Flight
from ...core.traffic import Traffic

# ---------------------- Model Architecture ----------------------

class ResidualGaussianBN(nn.Module):
    """
    Autoregressive Gaussian Bayesian Network trained on residuals.
    
    Predicts a diagonal Gaussian over the next-step residual:
        delta_t = y_t - y_{t-1}
    conditioned on (y_{t-1}, context).

    Inputs per step:
        prev_state: (B, 7)  = [x, y, z, vx, vy, vz, psi_rate]  (normalized)
        context:    (B, 8)  = [x0,y0,z0,cos,sin,gs_last,vz_last,psi_rate_last] (normalized)
    Outputs per step:
        mean_delta: (B, 7)
        log_std:    (B, 7)
    """
    def __init__(self, state_dim: int = 7, context_dim: int = 8, hidden: int = 512):
        super().__init__()
        in_dim = state_dim + context_dim
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, state_dim)
        self.log_std_head = nn.Linear(hidden, state_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, prev_state: torch.Tensor, context: torch.Tensor):
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if prev_state.dim() == 1:
            prev_state = prev_state.unsqueeze(0)
        # Repeat context across time if needed (caller aligns shapes)
        h = self.net(torch.cat([prev_state, context], dim=-1))
        mean = self.mean_head(h)
        # Small floor to avoid zero-variance pathologies; stays in normalized space
        log_std = self.log_std_head(h).clamp(min=-5.0, max=3.0)
        return mean, log_std

def load_model_checkpoint(checkpoint_path: str, device=None) -> tuple[ResidualGaussianBN, Optional[dict]]:
    """Load model checkpoint and return configured model with optional normalization stats.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on (auto-detects if None)

    Returns:
        Tuple of (Loaded ResidualGaussianBN instance, norm_stats dict or None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle potential module prefix issues
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}

    # Get model configuration from checkpoint or use defaults
    cfg = ckpt.get(
        "model_cfg",
        dict(
            state_dim=7,
            context_dim=8,
            hidden=512,
        ),
    )

    # Initialize and load model
    model = ResidualGaussianBN(**cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Extract normalization stats if available
    norm_stats = ckpt.get("norm_stats", None)

    return model, norm_stats

# ---------------------- Inference Utilities ----------------------

@torch.no_grad()
def bn_rollout(
    model: ResidualGaussianBN,
    last_hist_state: torch.Tensor,
    context: torch.Tensor,
    horizon: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Autoregressive rollout in normalized aircraft-centric space.
    
    Args:
        model: ResidualGaussianBN model
        last_hist_state: (B, 7) normalized final history token
        context: (B, 8) normalized context
        horizon: number of future steps (e.g., 12 for 60s at 5s stride)
        temperature: scales std during sampling (0 = deterministic means)
    
    Returns:
        futures_norm: (B, horizon, 7) normalized aircraft-centric trajectory
    """
    device = next(model.parameters()).device
    B = last_hist_state.size(0)
    prev = last_hist_state.to(device)
    ctx = context.to(device)
    traj = []
    for _ in range(horizon):
        mean_d, log_std = model(prev, ctx)
        if temperature <= 0:
            delta = mean_d
        else:
            std = torch.exp(log_std) * float(temperature)
            eps = torch.randn_like(std)
            delta = mean_d + eps * std
        nxt = prev + delta
        traj.append(nxt)
        prev = nxt
    return torch.stack(traj, dim=1)

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
    # Reshape to (1, 1, D) so they can be broadcasted across batch and time dimensions
    fm = torch.as_tensor(feat_mean, dtype=seq_norm.dtype, device=seq_norm.device).view(
        1, 1, -1
    )
    fs = torch.as_tensor(feat_std, dtype=seq_norm.dtype, device=seq_norm.device).view(
        1, 1, -1
    )

    # Denormalize the sequence using feature statistics: seq = seq_norm * std + mean
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
    # Use R^T (transpose/inverse of rotation matrix): [cos θ, sin θ; -sin θ, cos θ]
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


# ---------------------- BN Predictor Class ----------------------

class BNPredict:
    """Bayesian Network predictor for flight trajectories.

    This class implements trajectory prediction using a trained autoregressive
    Gaussian Bayesian Network model. It takes a Flight object with at least 60 seconds
    of historical data and generates probabilistic predictions of future
    trajectory segments.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        n_samples: int = 10,
        temperature: float = 1.0,
    ):
        """Initialize the BN predictor.

        Args:
            model_path: Path to the trained model checkpoint
            stats_path: Optional path to the normalization statistics JSON file.
                       If None, will try to load norm_stats from the checkpoint.
            device: Device to run inference on ('cuda', 'cpu', etc.)
            n_samples: Number of trajectory samples to generate
            temperature: Sampling temperature (0 = deterministic, >0 = stochastic)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model and stats
        self.model, checkpoint_norm_stats = load_model_checkpoint(model_path, self.device)
        
        if checkpoint_norm_stats is not None:
            self.norm_stats = checkpoint_norm_stats
        else:
            raise ValueError(
                "Normalization statistics not found in checkpoint"
            )

        self.n_samples = n_samples
        self.temperature = temperature

        # Extract stats
        self.feat_mean = np.array(self.norm_stats["feat_mean"], dtype=np.float32)
        self.feat_std = np.array(self.norm_stats["feat_std"], dtype=np.float32)
        self.ctx_mean = np.array(self.norm_stats["ctx_mean"], dtype=np.float32)
        self.ctx_std = np.array(self.norm_stats["ctx_std"], dtype=np.float32)

        # Extract training configuration from prep
        prep = self.norm_stats.get("prep", {})
        self.input_len = prep.get("input_len", 60)
        self.output_horizon = prep.get("output_horizon", 60)
        self.output_stride = prep.get("output_stride", 5)

    def _resample_predictions(
        self, predictions: torch.Tensor, T_out: int, sampling_rate: float
    ) -> torch.Tensor:
        """
        Resample predictions from model's native stride intervals to desired sampling rate.

        predictions: (n_samples, T_out, D) at output_stride intervals
        sampling_rate: new step in seconds. We resample onto [output_stride, output_stride+sampling_rate, ..., output_horizon].
        """
        assert predictions.dim() == 3, "predictions must be (S, T, D)"

        S, T, D = predictions.shape

        # Original prediction times based on output_stride and output_horizon
        original_times = np.arange(
            self.output_stride, self.output_horizon + 1, self.output_stride, dtype=float
        )  # [output_stride, 2*output_stride, ..., output_horizon]

        new_times = np.arange(
            self.output_stride, self.output_horizon + 1e-6, sampling_rate, dtype=float
        )  # [output_stride, output_stride+sampling_rate, ..., output_horizon]

        # Interpolate along time for each feature independently
        out = torch.empty(
            (S, len(new_times), D), dtype=predictions.dtype, device=predictions.device
        )

        pred_np = predictions.detach().cpu().numpy()  # (S,T,D)

        for f in range(D):
            vals = pred_np[:, :, f]  # (S,T)
            res = np.vstack(
                [np.interp(new_times, original_times, vals_i) for vals_i in vals]
            )  # (S, len(new_times))
            out[:, :, f] = torch.from_numpy(res).to(out.device, out.dtype)

        return out

    def preprocess_flight(self, flight: Flight) -> tuple[np.ndarray, pd.Timestamp]:
        """Preprocess flight data for model input.

        Args:
            flight: Flight object with trajectory data

        Returns:
            Tuple of (processed_features, last_timestamp)
        """
        # Get last input_len seconds of data
        window = flight.last(seconds=self.input_len)
        if window is None or len(window.data) < self.input_len:
            raise ValueError(f"Flight must have at least {self.input_len} seconds of data")

        df = window.data.copy()

        # Ensure proper column names and types
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Assume input is already properly sorted and resampled to exactly input_len samples at 1Hz
        if len(df) != self.input_len:
            raise ValueError(f"Expected exactly {self.input_len} samples, got {len(df)}")

        # Get last timestamp for prediction start
        last_timestamp = df["timestamp"].iloc[-1]

        # Convert coordinates to LV95
        import pyproj
        crs_lv95 = pyproj.CRS.from_epsg(2056)
        crs_wgs84 = pyproj.CRS.from_epsg(4326)
        to_lv95 = pyproj.Transformer.from_crs(crs_wgs84, crs_lv95, always_xy=True)

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
        df["psi_rate"] = np.nan_to_num(df["psi_rate"], nan=0.0, posinf=0.0, neginf=0.0)
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
            or Traffic object containing multiple predicted trajectories if n_samples>1
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
        C_norm = ((C_raw - self.ctx_mean[:len(C_raw)]) / self.ctx_std[:len(C_raw)]).astype(np.float32)

        # Convert to tensors
        x_hist = torch.from_numpy(X_norm).unsqueeze(0).to(self.device)
        ctx = torch.from_numpy(C_norm).unsqueeze(0).to(self.device)

        # Get last history state for autoregressive rollout
        last_hist_state = x_hist[:, -1, :]  # (1, 7)

        # Generate multiple samples by repeating inputs
        repeated_last_state = last_hist_state.repeat(self.n_samples, 1)
        repeated_ctx = ctx.repeat(self.n_samples, 1)

        # Calculate T_out from output_horizon and output_stride
        T_out = self.output_horizon // self.output_stride
        futures_norm = bn_rollout(
            self.model,
            repeated_last_state,
            repeated_ctx,
            horizon=T_out,
            temperature=self.temperature,
        )

        # Denormalize and convert back to global coordinates
        futures_global = denorm_seq_to_global(
            futures_norm, repeated_ctx, self.feat_mean, self.feat_std, self.ctx_mean, self.ctx_std
        )

        # Resample predictions from 5s intervals to 1s intervals for smoother output
        futures_global = self._resample_predictions(futures_global, T_out, sampling_rate=1)

        # Convert to lat/lon/alt
        futures_np = futures_global.detach().cpu().numpy()
        import pyproj
        crs_lv95 = pyproj.CRS.from_epsg(2056)
        crs_wgs84 = pyproj.CRS.from_epsg(4326)
        to_wgs84 = pyproj.Transformer.from_crs(crs_lv95, crs_wgs84, always_xy=True)

        predicted_flights = []
        for i in range(self.n_samples):
            lon, lat = to_wgs84.transform(futures_np[i, :, 0], futures_np[i, :, 1])
            alt_ft = futures_np[i, :, 2] / 0.3048

            # Create timestamps for predictions (1s intervals from last_timestamp + output_stride)
            # Predictions start at output_stride seconds (model's native prediction interval)
            pred_timestamps = [
                last_timestamp + timedelta(seconds=self.output_stride + j)
                for j in range(len(lon))  # Match the number of prediction points
            ]

            # Create flight data for this prediction
            flight_id = flight.flight_id if flight.flight_id is not None else (flight.callsign if flight.callsign is not None else "predicted")
            if self.n_samples > 1:
                # Add sample index to distinguish multiple predictions
                flight_id = f"{flight_id}_sample_{i}"

            pred_data = pd.DataFrame({
                "timestamp": pred_timestamps,
                "latitude": lat,
                "longitude": lon,
                "altitude": alt_ft,
                "flight_id": flight_id,
            })

            # Create flight with only predicted data (no historical data)
            predicted_flights.append(Flight(pred_data))

        # Return single Flight if n_samples=1, otherwise return Traffic
        if self.n_samples == 1:
            return predicted_flights[0]
        else:
            return Traffic.from_flights(predicted_flights)