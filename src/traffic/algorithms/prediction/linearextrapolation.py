import math
from datetime import timedelta
from typing import Any

from impunity import impunity

import pandas as pd

from ...core import types as tt
from ...core.flight import Flight
from . import PredictorBase


class LinearExtrapolation(PredictorBase):
    """Projects the trajectory in a straight line.

    The method uses the last known position, groundspeed (knots),
    vertical rate (ft/min), and track angle (degrees) to extrapolate
    forward in time assuming constant velocity in both horizontal
    and vertical directions.

    The prediction uses projected x/y coordinates (meters) for accurate
    distance calculations, avoiding issues with latitude/longitude
    spherical geometry.

    Parameters:
        duration: Duration to extrapolate (default: 1 minute)
        sampling_rate: Time between points in seconds (default: 1 second)
        forward: Legacy parameter, use duration instead
    """

    method_name = "linear_extrapolation"

    def __init__(
        self,
        duration: str | pd.Timedelta = "1 minute",
        sampling_rate: float = 1.0,
        forward: None | str | pd.Timedelta = None,
        **kwargs: Any,
    ):
        # Support legacy 'forward' parameter for backward compatibility
        if forward is not None:
            if isinstance(forward, str):
                self.duration = pd.Timedelta(forward)
            else:
                self.duration = forward
        else:
            if isinstance(duration, str):
                self.duration = pd.Timedelta(duration)
            else:
                self.duration = duration

        # Input validation
        if self.duration.total_seconds() <= 0:
            raise ValueError("Duration must be positive")
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")

        self.sampling_rate = sampling_rate

    @impunity(ignore_warnings=True)
    def predict(self, flight: Flight) -> Flight:
        last_line = flight.at()
        if last_line is None:
            raise ValueError("Unknown data for this flight")
        window = flight.last(seconds=20)

        if window is None:
            raise RuntimeError("Flight expect at least 20 seconds of data")

        # Use the last known values for linear extrapolation
        # Get the most recent data point (last position)
        last_data = window.data.iloc[-1]

        # Extract the last known values
        new_gs: tt.speed = last_data.groundspeed  # in knots
        if pd.isna(new_gs) or new_gs <= 0:
            raise ValueError("Invalid groundspeed data in flight trajectory")

        new_vr: tt.vertical_rate = last_data.vertical_rate  # in ft/min
        if pd.isna(new_vr):
            new_vr = 0.0  # Default to level flight if vertical rate is unknown

        # Convert groundspeed from knots to m/s for x/y calculations
        # 1 knot = 0.514444 m/s (nautical mile/hour to meters/second)
        gs_ms = new_gs * 0.514444

        # Compute velocity components in x/y space
        track_rad = math.radians(last_line.track)
        vx = gs_ms * math.sin(track_rad)  # m/s in x direction
        vy = gs_ms * math.cos(track_rad)  # m/s in y direction

        # Project flight to x/y coordinates (meters) for easier calculation
        # Use a consistent projection for accurate round-trip conversion
        # This avoids issues with latitude/longitude spherical geometry
        projection = flight.projection(proj="lcc")  # Lambert Conformal Conical
        flight_xy = flight.compute_xy(projection=projection)

        # Get the last x/y position
        last_xy = flight_xy.at()
        if (
            last_xy is None
            or not hasattr(last_xy, "x")
            or not hasattr(last_xy, "y")
        ):
            raise ValueError("Could not compute x/y coordinates for flight")

        # Generate timestamps at the specified sampling rate
        total_seconds = self.duration.total_seconds()
        n_points = (
            int(total_seconds / self.sampling_rate) + 1
        )  # +1 to include the endpoint

        timestamps = [
            last_line.timestamp + timedelta(seconds=i * self.sampling_rate)
            for i in range(n_points)
        ]

        # Calculate positions for each timestamp
        data_points = []
        for i, ts in enumerate(timestamps):
            dt = (ts - last_line.timestamp).total_seconds()

            if i == 0:
                # First point is the last known position
                data_points.append(last_line)
            else:
                # Calculate new position in x/y space (much simpler!)
                new_x = last_xy.x + vx * dt
                new_y = last_xy.y + vy * dt

                # Calculate new altitude with bounds checking
                # vertical_rate is in ft/min, convert to ft/s by dividing by 60
                new_alt: tt.altitude = last_line.altitude + (new_vr * dt) / 60

                # Apply reasonable bounds to prevent unrealistic altitudes
                # Don't go below 0 ft or above 60,000 ft
                new_alt = max(0.0, min(60000.0, new_alt))

                # Create a temporary flight with the new x/y position
                # to convert back to lat/lon
                temp_data = pd.DataFrame(
                    [
                        {
                            "timestamp": ts,
                            "x": new_x,
                            "y": new_y,
                            "altitude": new_alt,
                            "groundspeed": new_gs,
                            "vertical_rate": new_vr,
                        }
                    ]
                )

                # Convert back to lat/lon using the same projection
                temp_flight = Flight(temp_data)
                temp_flight_latlon = temp_flight.compute_latlon_from_xy(
                    projection=projection
                )

                # Extract the lat/lon values
                latlon_data = temp_flight_latlon.data.iloc[0]
                new_lat = latlon_data.latitude
                new_lon = latlon_data.longitude

                data_points.append(
                    pd.Series(
                        {
                            "timestamp": ts,
                            "latitude": new_lat,
                            "longitude": new_lon,
                            "altitude": new_alt,
                            "groundspeed": new_gs,
                            "vertical_rate": new_vr,
                        }
                    )
                )

        return Flight(pd.DataFrame.from_records(data_points).ffill())
