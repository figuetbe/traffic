from datetime import timedelta
from typing import Any

from impunity import impunity
from pitot import geodesy as geo

import pandas as pd

from ...core import types as tt
from ...core.flight import Flight


class StraightLinePredict:
    """Projects the trajectory in a straight line.

    The method uses the last position of a trajectory and extrapolates
    forward in time using constant speed and heading.

    Parameters:
        duration: Duration to extrapolate (default: 1 minute)
        sampling_rate: Time between points in seconds (default: 1 second)
        forward: Legacy parameter, use duration instead
    """

    def __init__(
        self,
        duration: str | pd.Timedelta = "1 minute",
        sampling_rate: float = 1.0,
        forward: None | str | pd.Timedelta = None,
        **kwargs: Any
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

        self.sampling_rate = sampling_rate

    @impunity(ignore_warnings=True)
    def predict(self, flight: Flight) -> Flight:
        last_line = flight.at()
        if last_line is None:
            raise ValueError("Unknown data for this flight")
        window = flight.last(seconds=20)

        if window is None:
            raise RuntimeError("Flight expect at least 20 seconds of data")

        # Use average speed and vertical rate from the last 20 seconds
        new_gs: tt.speed = window.data.groundspeed.mean()
        new_vr: tt.vertical_rate = window.data.vertical_rate.mean()

        # Generate timestamps at the specified sampling rate
        total_seconds = self.duration.total_seconds()
        n_points = int(total_seconds / self.sampling_rate) + 1  # +1 to include the endpoint

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
                # Calculate new position using great circle distance
                new_lat, new_lon, _ = geo.destination(
                    last_line.latitude,
                    last_line.longitude,
                    last_line.track,
                    new_gs * dt,
                )

                # Calculate new altitude
                new_alt: tt.altitude = last_line.altitude + new_vr * dt

                data_points.append(pd.Series({
                    "timestamp": ts,
                    "latitude": new_lat,
                    "longitude": new_lon,
                    "altitude": new_alt,
                    "groundspeed": new_gs,
                    "vertical_rate": new_vr,
                }))

        return Flight(pd.DataFrame.from_records(data_points).ffill())
