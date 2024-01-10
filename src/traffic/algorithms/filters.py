from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,  # for python 3.8 and impunity
    Generic,
    Protocol,
    Type,
    TypedDict,
    TypeVar,
    cast,
)

from impunity import impunity
from scipy import signal
from typing_extensions import Annotated, NotRequired

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj

if TYPE_CHECKING:
    from cartopy import crs


class Filter(Protocol):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        ...


class FilterBase(Filter):
    tracked_variables: dict[str, list[Any]]
    projection: None | "crs.Projection" | pyproj.Proj = None

    def __or__(self, other: FilterBase) -> FilterBase:
        """Composition operator on filters."""

        def composition(
            object: Type[FilterBase], data: pd.DataFrame
        ) -> pd.DataFrame:
            return other.apply(self.apply(data))

        CombinedFilter = type(
            "CombinedFilter",
            (FilterBase,),
            dict(apply=composition),
        )
        return cast(FilterBase, CombinedFilter())


V = TypeVar("V")


class TrackVariable(Generic[V]):
    def __set_name__(self, owner: FilterBase, name: str) -> None:
        self.public_name = name

        if not hasattr(owner, "tracked_variables"):
            owner.tracked_variables = {}

        owner.tracked_variables[self.public_name] = []

    def __get__(
        self, obj: FilterBase, objtype: None | Type[FilterBase] = None
    ) -> V:
        history = cast(list[V], obj.tracked_variables[self.public_name])
        return history[-1]

    def __set__(self, obj: FilterBase, value: V) -> None:
        obj.tracked_variables[self.public_name].append(value)


class FilterMedian(FilterBase):
    """Rolling median filter"""

    # default kernel values
    default: ClassVar[dict[str, int]] = dict(
        altitude=11,
        geoaltitude=9,
        vertical_rate=5,
        groundspeed=9,
        track=5,
    )

    def __init__(self, **kwargs: int) -> None:
        self.columns = {**self.default, **kwargs}

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        for column, kernel in self.columns.items():
            if column not in data.columns:
                continue
            column_copy = data[column]
            data[column] = data[column].rolling(kernel, center=True).median()
            data.loc[data[column].isnull(), column] = column_copy
        return data


class FilterMean(FilterBase):
    """Rolling mean filter."""

    # default kernel values
    default: ClassVar[dict[str, int]] = dict(
        altitude=10,
        geoaltitude=10,
        vertical_rate=3,
        groundspeed=7,
        track=3,
    )

    def __init__(self, **kwargs: int) -> None:
        self.columns = {**self.default, **kwargs}

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        for column, kernel in self.columns.items():
            if column not in data.columns:
                continue
            column_copy = data[column]
            data[column] = data[column].rolling(kernel, center=True).mean()
            data.loc[data[column].isnull(), column] = column_copy
        return data


class FilterAboveSigmaMedian(FilterBase):
    """Filters noisy values above one sigma wrt median filter.

    The method first applies a median filter on each feature of the
    DataFrame. A default kernel size is applied for a number of features
    (resp. latitude, longitude, altitude, track, groundspeed, IAS, TAS) but
    other kernel values may be passed as kwargs parameters.

    Rather than returning averaged values, the method computes thresholds
    on sliding windows (as an average of squared differences) and replace
    unacceptable values with NaNs.

    Then, a strategy may be applied to fill the NaN values, by default a
    forward/backward fill. Other strategies may be passed, for instance *do
    nothing*: ``None``; or *interpolate*: ``lambda x: x.interpolate()``.

    .. note::

        This method if often more efficient when applied several times with
        different kernel values.Kernel values may be passed as integers, or
        list/tuples of integers for cascade of filters:

        .. code:: python

            # this cascade of filters appears to work well on altitude
            flight.filter(altitude=17).filter(altitude=53)

            # this is equivalent to the default value
            flight.filter(altitude=(17, 53))

    """

    # default kernel values
    default: ClassVar[dict[str, int | tuple[int, ...]]] = dict(
        altitude=(17, 53),
        geoaltitude=(17, 53),
        selected_mcp=(17, 53),
        selected_fms=(17, 53),
        IAS=23,
        TAS=23,
        Mach=23,
        vertical_rate=3,
        groundspeed=5,
        compute_gs=(17, 53),
        compute_track=17,
        track=17,
        onground=3,
    )

    def __init__(self, **kwargs: int | tuple[int, ...]) -> None:
        self.columns = {**self.default, **kwargs}
        self.empty_kwargs = len(kwargs) == 0

    def cascaded_filters(
        self,
        df: pd.DataFrame,
        feature: str,
        kernel_size: int,
        filt: Callable[[pd.Series, int], Any] = signal.medfilt,
    ) -> pd.DataFrame:
        """Produces a mask for data to be discarded.

        The filtering applies a low pass filter (e.g medfilt) to a signal
        and measures the difference between the raw and the filtered signal.

        The average of the squared differences is then produced (sq_eps) and
        used as a threshold for filtering.

        Errors may raised if the kernel_size is too large
        """
        y = df[feature].astype(float)
        y_m = filt(y, kernel_size)
        sq_eps = (y - y_m) ** 2
        return pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "y": y,
                "y_m": y_m,
                "sq_eps": sq_eps,
                "sigma": np.sqrt(filt(sq_eps, kernel_size)),
            },
            index=df.index,
        )

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = self.columns
        if self.empty_kwargs:
            features = [
                cast(str, feature)
                for feature in data.columns
                if data[feature].dtype
                in [np.float32, np.float64, np.int32, np.int64]
            ]
            columns = {
                **dict((feature, 17) for feature in features),
                **columns,
            }

        for column, kernels in columns.items():
            if column not in data.columns:
                continue

            if isinstance(kernels, int):
                kernels = (kernels,)
            else:
                kernels = tuple(kernels)

            for size in kernels:
                # Prepare each feature for the filtering
                df = self.cascaded_filters(
                    data[["timestamp", column]], column, size
                )

                # Decision to accept/reject data points in the time series
                data.loc[df.sq_eps > df.sigma, column] = None

        return data


class DerivativeParams(TypedDict):
    first: float  # threshold for 1st derivative
    second: float  # threshold for 2nd derivative
    kernel: int


class FilterDerivative(FilterBase):
    """Filter based on the 1st and 2nd derivatives of parameters

    The method computes the absolute value of the 1st and 2nd derivatives
    of the parameters. If the value of the derivatives is above the defined
    threshold values, the datapoint is removed

    """

    # default parameter values
    default: ClassVar[dict[str, DerivativeParams]] = dict(
        altitude=dict(first=200, second=150, kernel=10),
        geoaltitude=dict(first=200, second=150, kernel=10),
        vertical_rate=dict(first=1500, second=1000, kernel=5),
        groundspeed=dict(first=12, second=10, kernel=3),
        track=dict(first=12, second=10, kernel=2),
    )

    def __init__(
        self, time_column: str = "timestamp", **kwargs: DerivativeParams
    ) -> None:
        """

        :param time_column: the name of the time column (default: "timestamp")

        :param kwargs: each keyword argument has the name of a feature.
            the value must be a dictionary with the following keys:
            - first: threshold value for the first derivative
            - second: threshold value for the second derivative
            - kernel: the kernel size in seconds

        If two spikes are detected within the width of the kernel, all
        datapoints inbetween are also removed.

        """
        self.columns = {**self.default, **kwargs}
        self.time_column = time_column

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        timediff = data[self.time_column].diff().dt.total_seconds()
        for column, params in self.columns.items():
            if column not in data.columns:
                continue
            window = params["kernel"]
            diff1 = data[column].diff().abs()
            # TODO it could be smarter to use the unwrapped version...
            if column == "track":
                diff1.loc[(diff1 < 370) & (diff1 > 350)] = 0
            diff2 = diff1.diff().abs()

            deriv1 = diff1 / timediff
            deriv2 = diff2 / timediff
            spike = np.bitwise_or(
                (deriv1 >= params["first"]), (deriv2 >= params["second"])
            ).fillna(False, inplace=False)

            spike_time = pd.Series(np.nan, index=data.index)
            spike_time.loc[spike] = data[self.time_column].loc[spike]

            if not spike_time.isnull().all():
                spike_time_prev = spike_time.ffill()
                spike_delta_prev = data["timestamp"] - spike_time_prev
                spike_time_next = spike_time.bfill()
                spike_delta_next = spike_time_next - data["timestamp"]
                in_window = np.bitwise_and(
                    spike_delta_prev.dt.total_seconds() <= window,
                    spike_delta_next.dt.total_seconds() <= window,
                )
                data.loc[(in_window), column] = np.NaN

        return data


class ClusteringParams(TypedDict):
    group_size: int
    value_threshold: float
    time_threshold: NotRequired[float]


class FilterClustering(FilterBase):
    """Filter based on clustering.

    The method creates clusters of datapoints based on the difference in time
    and parameter value. If the cluster is larger than the defined group size
    the datapoints are kept, otherwise they are removed.

    """

    default: ClassVar[dict[str, ClusteringParams]] = dict(
        altitude=dict(group_size=15, value_threshold=500),
        geoaltitude=dict(group_size=15, value_threshold=500),
        vertical_rate=dict(group_size=15, value_threshold=500),
        onground=dict(group_size=15, value_threshold=500),
        track=dict(group_size=15, value_threshold=500),
        latitude=dict(group_size=15, value_threshold=500),
        longitude=dict(group_size=15, value_threshold=500),
    )

    def __init__(
        self, time_column: str = "timestamp", **kwargs: ClusteringParams
    ) -> None:
        """
        :param time_column: the name of the time column (default: "timestamp")

        :param kwargs: each keyword argument has the name of a feature.
            the value must be a dictionary with the following keys:
            - group_size: minimum size of the cluster to be kept
            - value_threshold: within the value threshold, the samples fall in
              the same cluster
            - time_threshold: within the time threshold, the samples fall in
              the same cluster
        """
        self.columns = {**self.default, **kwargs}
        self.time_column = time_column

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop_duplicates([self.time_column], keep="last")

        for column, param in self.columns.items():
            if column not in data.columns:
                continue
            if column == "onground":
                statechange = data[column].diff().astype(bool)
                data["group"] = statechange.eq(True).cumsum()
                groups = data["group"].value_counts()
                keepers = groups[groups > param["group_size"]].index.tolist()
                data[column] = data[column].where(
                    data["group"].isin(keepers), float("NaN")
                )

            else:
                timediff = data[self.time_column].diff().dt.total_seconds()
                temp_index = data.index
                temp_values = data[column].dropna()
                paradiff = temp_values.diff().reindex(temp_index).abs()
                bigdiff = pd.Series(
                    np.bitwise_or(
                        (timediff > param.get("time_threshold", 60)),
                        (paradiff > param["value_threshold"]),
                    )
                )
                data["group"] = bigdiff.eq(True).cumsum()
                groups = data[data[column].notna()]["group"].value_counts()
                keepers = groups[groups > param["group_size"]].index.tolist()
                data[column] = data[column].where(
                    data["group"].isin(keepers), float("NaN")
                )

        return data.drop(columns=["group"])


class ProcessXYFilterBase(FilterBase):
    @impunity
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        groundspeed: Annotated[Any, "kts"] = df.groundspeed
        track: Annotated[Any, "radians"] = np.radians(90 - df.track)

        velocity: Annotated[Any, "m/s"] = groundspeed

        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y

        dx: Annotated[Any, "m/s"] = velocity * np.cos(track)
        dy: Annotated[Any, "m/s"] = velocity * np.sin(track)

        return pd.DataFrame({"x": x, "y": y, "dx": dx, "dy": dy})

    @impunity
    def postprocess(
        self, df: pd.DataFrame
    ) -> Dict[str, npt.NDArray[np.float64]]:
        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y

        dx: Annotated[Any, "m/s"] = df.dx
        dy: Annotated[Any, "m/s"] = df.dy

        velocity: Annotated[Any, "m/s"] = np.sqrt(dx**2 + dy**2)
        track = 90 - np.degrees(np.arctan2(dy, dx))
        track = track % 360
        groundspeed: Annotated[Any, "kts"] = velocity

        return dict(
            x=x,
            y=y,
            groundspeed=groundspeed,
            track=track,
        )


class ProcessXYZFilterBase(FilterBase):
    @impunity
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        alt: Annotated[Any, "ft"] = df.altitude
        groundspeed: Annotated[Any, "kts"] = df.groundspeed
        track: Annotated[Any, "radians"] = np.radians(90 - df.track)
        vertical_rate: Annotated[Any, "ft/min"] = df.vertical_rate

        velocity: Annotated[Any, "m/s"] = groundspeed

        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y
        z: Annotated[Any, "m"] = alt

        dx: Annotated[Any, "m/s"] = velocity * np.cos(track)
        dy: Annotated[Any, "m/s"] = velocity * np.sin(track)
        dz: Annotated[Any, "m/s"] = vertical_rate

        return pd.DataFrame(
            {"x": x, "y": y, "z": z, "dx": dx, "dy": dy, "dz": dz}
        )

    @impunity
    def postprocess(
        self, df: pd.DataFrame
    ) -> Dict[str, npt.NDArray[np.float64]]:
        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y
        z: Annotated[Any, "m"] = df.z

        dx: Annotated[Any, "m/s"] = df.dx
        dy: Annotated[Any, "m/s"] = df.dy
        dz: Annotated[Any, "m/s"] = df.dz

        velocity: Annotated[Any, "m/s"] = np.sqrt(dx**2 + dy**2)
        track = 90 - np.degrees(np.arctan2(dy, dx))
        track = track % 360

        altitude: Annotated[Any, "ft"] = z
        groundspeed: Annotated[Any, "kts"] = velocity
        vertical_rate: Annotated[Any, "ft/min"] = dz

        return dict(
            x=x,
            y=y,
            altitude=altitude,
            groundspeed=groundspeed,
            track=track,
            vertical_rate=vertical_rate,
        )


class KalmanFilter6D(ProcessXYZFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p_pre: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    def __init__(self, reject_sigma: int = 3) -> None:
        super().__init__()
        self.reject_sigma = reject_sigma

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(data)

        for cumul in self.tracked_variables.values():
            cumul.clear()

        # initial state
        _id6 = np.eye(df.shape[1])
        dt = 1

        self.x_mes = df.iloc[0].values
        self.x_cor = df.iloc[0].values
        self.p_pre = _id6 * 1e5
        self.p_cor = _id6 * 1e5

        R = (
            np.diag(
                [
                    (df.x - df.x.rolling(17).mean()).std(),
                    (df.y - df.y.rolling(17).mean()).std(),
                    (df.z - df.z.rolling(17).mean()).std(),
                    (df.dx - df.dx.rolling(17).mean()).std(),
                    (df.dy - df.dy.rolling(17).mean()).std(),
                    (df.dz - df.dz.rolling(17).mean()).std(),
                ]
            )
            ** 2
        )

        # plus de bruit de modèle sur les dérivées qui sont pas recalées
        Q = 1e-1 * np.diag([0.25, 0.25, 0.25, 1, 1, 1]) * R
        Q += 0.5 * np.diag(Q.diagonal()[3:], k=3)
        Q += 0.5 * np.diag(Q.diagonal()[3:], k=-3)

        for i in range(1, df.shape[0]):
            # measurement
            self.x_mes = df.iloc[i].values
            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            A = np.eye(6) + dt * np.eye(6, k=3)
            x_pre = A @ self.x_cor
            p_pre = A @ self.p_cor @ A.T + Q
            H = _id6.copy()

            # DEBUG: p_pre should be symmetric
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Logging the final value...
            self.p_pre = p_pre

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id6 - K @ H
            self.p_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

            # DEBUG: p_cor should be symmetric
            # assert np.abs(self.p_cor - self.p_cor.T).sum() < 1e-6

        filtered = pd.DataFrame(
            self.tracked_variables["x_cor"],
            columns=["x", "y", "z", "dx", "dy", "dz"],
        )

        return data.assign(**self.postprocess(filtered))


class ProcessXYZZFilterBase(FilterBase):
    @impunity
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        groundspeed: Annotated[Any, "kts"] = df.groundspeed
        # the angle is wrapped between 0 and 2pi, we need to unwrap it
        math_angle: Annotated[Any, "radians"] = np.unwrap(
            np.radians(90 - df.track)
        )
        velocity: Annotated[Any, "m/s"] = groundspeed

        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y

        altitude: Annotated[Any, "ft"] = df.altitude
        geoaltitude: Annotated[Any, "ft"] = df.geoaltitude
        alt_baro: Annotated[Any, "m"] = altitude
        alt_geo: Annotated[Any, "m"] = geoaltitude

        vertical_rate: Annotated[Any, "ft/min"] = df.vertical_rate
        vert_rate: Annotated[Any, "m/s"] = vertical_rate

        return pd.DataFrame(
            {
                "x": x,
                "y": y,
                "alt_baro": alt_baro,
                "alt_geo": alt_geo,
                "math_angle": math_angle,
                "velocity": velocity,
                "vert_rate": vert_rate,
            }
        )

    @impunity
    def postprocess(
        self, df: pd.DataFrame
    ) -> Dict[str, npt.NDArray[np.float64]]:
        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y
        alt_baro: Annotated[Any, "m"] = df.alt_baro
        alt_geo: Annotated[Any, "m"] = df.alt_geo
        math_angle: Annotated[Any, "radians"] = df.math_angle
        velocity: Annotated[Any, "m/s"] = df.velocity
        vert_rate: Annotated[Any, "m/s"] = df.vert_rate

        altitude: Annotated[Any, "ft"] = alt_baro
        geoaltitude: Annotated[Any, "ft"] = alt_geo
        track: Annotated[Any, "degree"] = (90 - np.degrees(math_angle)) % 360
        groundspeed: Annotated[Any, "kts"] = velocity
        vertical_rate: Annotated[Any, "ft/min"] = vert_rate

        return dict(
            x=x,
            y=y,
            altitude=altitude,
            geoaltitude=geoaltitude,
            track=track,
            groundspeed=groundspeed,
            vertical_rate=vertical_rate,
        )


class EKF7D(ProcessXYZZFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p_pre: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    def state_transition(self, dt: float) -> npt.NDArray[np.float64]:
        # Unpack the state vector
        _, _, alt_baro, alt_geo, math_angle, velocity, vert_rate = self.x_cor

        # Compute the derivatives
        x_dot = velocity * np.cos(math_angle)
        y_dot = velocity * np.sin(math_angle)
        alt_baro_dot = vert_rate
        alt_geo_dot = vert_rate

        # Compute the predicted state
        x_pred = self.x_cor.copy()
        x_pred[0] += x_dot * dt
        x_pred[1] += y_dot * dt
        x_pred[2] += alt_baro_dot * dt
        x_pred[3] += alt_geo_dot * dt
        # math_angle, velocity, vert_rate) assumed to be constant over the dt
        return x_pred

    def update_A_jacobian(
        self, x_pre: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        # Unpack the state vector
        _, _, _, _, math_angle, velocity, vert_rate = x_pre

        # Compute the Jacobian matrix
        A_jacobian = np.eye(7)
        A_jacobian[0, 4] = (
            -velocity * np.sin(math_angle) * dt
        )  # Partial derivative of x_dot w.r.t math_angle
        A_jacobian[0, 5] = (
            np.cos(np.radians(math_angle)) * dt
        )  # Partial derivative of x_dot w.r.t groundspeed
        A_jacobian[1, 4] = (
            -velocity * np.sin(math_angle) * dt
        )  # Partial derivative of y_dot w.r.t math_angle
        A_jacobian[1, 5] = (
            np.sin(np.radians(math_angle)) * dt
        )  # Partial derivative of y_dot w.r.t groundspeed
        A_jacobian[
            2, 6
        ] = dt  # Partial derivative of altitude_dot w.r.t vert_rate
        A_jacobian[
            3, 6
        ] = dt  # Partial derivative of geoaltitude_dot w.r.t vert_rate

        return A_jacobian

    def __init__(self, reject_sigma: int = 3) -> None:
        super().__init__()
        self.reject_sigma = reject_sigma

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(data)

        for cumul in self.tracked_variables.values():
            cumul.clear()

        # initial state
        _id7 = np.eye(df.shape[1])

        self.x_mes = df.iloc[0].values
        self.x_cor = df.iloc[0].values
        self.p_pre = _id7 * 1e5
        self.p_cor = _id7 * 1e5

        R = (
            np.diag(
                [
                    (df.x - df.x.rolling(17).mean()).std(),
                    (df.y - df.y.rolling(17).mean()).std(),
                    (df.alt_baro - df.alt_baro.rolling(17).mean()).std(),
                    (df.alt_geo - df.alt_geo.rolling(17).mean()).std(),
                    (df.math_angle - df.math_angle.rolling(17).mean()).std(),
                    (df.velocity - df.velocity.rolling(17).mean()).std(),
                    (df.vert_rate - df.vert_rate.rolling(17).mean()).std(),
                ]
            )
            ** 2
        )

        # plus de bruit de modèle sur les dérivées qui sont pas recalées
        Q = 1e-1 * np.diag([0.25, 0.25, 0.25, 0.25, 1, 1, 1]) * R
        Q += 0.5 * np.diag(Q.diagonal()[4:], k=4)
        Q += 0.5 * np.diag(Q.diagonal()[4:], k=-4)

        for i in range(1, df.shape[0]):
            # measurement
            dt = (
                data.iloc[i]["timestamp"] - data.iloc[i - 1]["timestamp"]
            ).total_seconds()
            self.x_mes = df.iloc[i].values
            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            # Use the state_transition function to predict the next state
            x_pre = self.state_transition(dt)
            # Calculate the Jacobian of the state transition function
            A = self.update_A_jacobian(x_pre, dt)
            p_pre = A @ self.p_cor @ A.T + Q
            H = _id7.copy()

            # DEBUG: p_pre should be symmetric
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Logging the final value...
            self.p_pre = p_pre

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id7 - K @ H
            self.p_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

            # DEBUG: p_cor should be symmetric
            # assert np.abs(self.p_cor - self.p_cor.T).sum() < 1e-6

        filtered = pd.DataFrame(
            self.tracked_variables["x_cor"],
            columns=[
                "x",
                "y",
                "alt_baro",
                "alt_geo",
                "math_angle",
                "velocity",
                "vert_rate",
            ],
        )

        return data.assign(**self.postprocess(filtered))


class KalmanSmoother6D(ProcessXYZFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    xs: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    def __init__(self, reject_sigma: int = 3) -> None:
        super().__init__()
        self.reject_sigma = reject_sigma

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(data)

        for cumul in self.tracked_variables.values():
            cumul.clear()

        # initial state
        _id6 = np.eye(df.shape[1])

        self.x_mes = df.iloc[0].values
        self.x1_cor = df.iloc[0].values
        self.p1_cor = _id6 * 1e5

        R = (
            np.diag(
                [
                    (df.x - df.x.rolling(17).mean()).std(),
                    (df.y - df.y.rolling(17).mean()).std(),
                    (df.z - df.z.rolling(17).mean()).std(),
                    (df.dx - df.dx.rolling(17).mean()).std(),
                    (df.dy - df.dy.rolling(17).mean()).std(),
                    (df.dz - df.dz.rolling(17).mean()).std(),
                ]
            )
            ** 2
        )

        # plus de bruit de modèle sur les dérivées qui sont pas recalées
        Q = 1e-1 * np.diag([0.25, 0.25, 0.25, 1, 1, 1]) * R
        Q += 0.5 * np.diag(Q.diagonal()[3:], k=3)
        Q += 0.5 * np.diag(Q.diagonal()[3:], k=-3)

        # >>> First FORWARD  <<<

        for i in range(1, df.shape[0]):
            dt = (
                data.iloc[i]["timestamp"] - data.iloc[i - 1]["timestamp"]
            ).total_seconds()
            # measurement
            self.x_mes = df.iloc[i].values

            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            A = np.eye(6) + dt * np.eye(6, k=3)
            x_pre = A @ self.x1_cor
            p_pre = A @ self.p1_cor @ A.T + Q
            H = _id6.copy()

            # DEBUG symmetric matrices
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                # 3 sigma ^2 * nb_dim (6)

                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p1_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x1_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id6 - K @ H
            self.p1_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

            # DEBUG symmetric matrices
            # assert np.abs(self.p1_cor - self.p1_cor.T).sum() < 1e-6

        # >>> Now BACKWARD  <<<
        self.x2_cor = self.x1_cor
        self.p2_cor = 100 * self.p1_cor

        for i in range(df.shape[0] - 1, 0, -1):
            # measurement
            dt = (
                data.iloc[i - 1]["timestamp"] - data.iloc[i]["timestamp"]
            ).total_seconds()
            self.x_mes = df.iloc[i].values
            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            A = np.eye(6) + dt * np.eye(6, k=3)
            x_pre = A @ self.x2_cor
            p_pre = A @ self.p2_cor @ A.T + Q
            H = _id6.copy()

            # DEBUG symmetric matrices
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                # 3 sigma ^2 * nb_dim (6)

                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p2_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Logging the final value...
            self.p_pre = p_pre

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x2_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id6 - K @ H
            self.p2_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

            # DEBUG symmetric matrices
            # assert np.abs(self.p2_cor - self.p2_cor.T).sum() < 1e-6

        # and the smoothing
        x1_cor = np.array(self.tracked_variables["x1_cor"])
        p1_cor = np.array(self.tracked_variables["p1_cor"])
        x2_cor = np.array(self.tracked_variables["x2_cor"][::-1])
        p2_cor = np.array(self.tracked_variables["p2_cor"][::-1])

        for i in range(1, df.shape[0]):
            s1 = np.linalg.inv(p1_cor[i])
            s2 = np.linalg.inv(p2_cor[i])
            self.ps = np.linalg.inv(s1 + s2)
            self.xs = (self.ps @ (s1 @ x1_cor[i] + s2 @ x2_cor[i])).T

        filtered = pd.DataFrame(
            self.tracked_variables["xs"],
            columns=["x", "y", "z", "dx", "dy", "dz"],
        )

        return data.assign(**self.postprocess(filtered))


class EKFSmoother7D(ProcessXYZZFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    xs: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    def __init__(self, reject_sigma: int = 3) -> None:
        super().__init__()
        self.reject_sigma = reject_sigma

    def state_transition(
        self, x: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        # Unpack the state vector
        _, _, alt_baro, alt_geo, math_angle, velocity, vert_rate = x

        # Compute the derivatives
        x_dot = velocity * np.cos(math_angle)
        y_dot = velocity * np.sin(math_angle)
        alt_baro_dot = vert_rate
        alt_geo_dot = vert_rate

        # Compute the predicted state
        x_pred = x.copy()
        x_pred[0] += x_dot * dt
        x_pred[1] += y_dot * dt
        x_pred[2] += alt_baro_dot * dt
        x_pred[3] += alt_geo_dot * dt
        # math_angle, velocity, vert_rate) assumed to be constant over the dt
        return x_pred

    def update_A_jacobian(
        self, x_cor: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        # Unpack the state vector
        _, _, _, _, math_angle, velocity, vert_rate = x_cor

        # Compute the Jacobian matrix
        A_jacobian = np.eye(7)
        A_jacobian[0, 4] = (
            -velocity * np.sin(math_angle) * dt
        )  # Partial derivative of x_dot w.r.t math_angle
        A_jacobian[0, 5] = (
            np.cos(np.radians(math_angle)) * dt
        )  # Partial derivative of x_dot w.r.t groundspeed
        A_jacobian[1, 4] = (
            -velocity * np.sin(math_angle) * dt
        )  # Partial derivative of y_dot w.r.t math_angle
        A_jacobian[1, 5] = (
            np.sin(np.radians(math_angle)) * dt
        )  # Partial derivative of y_dot w.r.t groundspeed
        A_jacobian[
            2, 6
        ] = dt  # Partial derivative of altitude_dot w.r.t vert_rate
        A_jacobian[
            3, 6
        ] = dt  # Partial derivative of geoaltitude_dot w.r.t vert_rate

        return A_jacobian

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(data)

        for cumul in self.tracked_variables.values():
            cumul.clear()

        # initial state
        _id7 = np.eye(df.shape[1])

        self.x_mes = df.iloc[0].values
        self.x1_cor = df.iloc[0].values
        self.p1_cor = _id7 * 1e5

        R = (
            np.diag(
                [
                    (df.x - df.x.rolling(17).mean()).std(),
                    (df.y - df.y.rolling(17).mean()).std(),
                    (df.alt_baro - df.alt_baro.rolling(17).mean()).std(),
                    (df.alt_geo - df.alt_geo.rolling(17).mean()).std(),
                    (df.math_angle - df.math_angle.rolling(17).mean()).std(),
                    (df.velocity - df.velocity.rolling(17).mean()).std(),
                    (df.vert_rate - df.vert_rate.rolling(17).mean()).std(),
                ]
            )
            ** 2
        )
        # R = np.diag(
        #     [
        #             6**2,  # x - GPS position [m] 1
        #             6**2,  # y - GPS position [m] 2
        #             20**2,  # altitude [m] 3
        #             20**2,  # geoaltitude [m] 4
        #             10*1.5**2,  # track [deg]
        #             1.5**2,  # groundspeed [m/s]
        #             1**2,  # vertical_rate [m/s] 7
        #         ]
        # )

        # plus de bruit de modèle sur les dérivées qui sont pas recalées
        Q = 1e-1 * np.diag([0.25, 0.25, 0.25, 0.25, 1, 1, 1]) * R
        Q += 0.5 * np.diag(Q.diagonal()[4:], k=4)
        Q += 0.5 * np.diag(Q.diagonal()[4:], k=-4)

        # >>> First FORWARD  <<<

        for i in range(1, df.shape[0]):
            dt = (
                data.iloc[i]["timestamp"] - data.iloc[i - 1]["timestamp"]
            ).total_seconds()
            # measurement
            self.x_mes = df.iloc[i].values

            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            # Use the state_transition function to predict the next state
            x_pre = self.state_transition(self.x1_cor, dt)
            # Calculate the Jacobian of the state transition function
            A = self.update_A_jacobian(x_pre, dt)
            p_pre = A @ self.p1_cor @ A.T + Q
            H = _id7.copy()

            # DEBUG symmetric matrices
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                # 3 sigma ^2 * nb_dim (6)

                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p1_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x1_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id7 - K @ H
            self.p1_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

            # DEBUG symmetric matrices
            # assert np.abs(self.p1_cor - self.p1_cor.T).sum() < 1e-6

        # >>> Now BACKWARD  <<<
        self.x2_cor = self.x1_cor
        self.p2_cor = 100 * self.p1_cor

        for i in range(df.shape[0] - 1, 0, -1):
            dt = (
                data.iloc[i - 1]["timestamp"] - data.iloc[i]["timestamp"]
            ).total_seconds()
            # measurement
            self.x_mes = df.iloc[i].values
            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            # A = np.eye(6) + dt * np.eye(6, k=3)
            # x_pre = A @ self.x2_cor
            x_pre = self.state_transition(self.x2_cor, dt)
            A = self.update_A_jacobian(x_pre, dt)
            p_pre = A @ self.p2_cor @ A.T + Q
            H = _id7.copy()

            # DEBUG symmetric matrices
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                # 3 sigma ^2 * nb_dim (6)

                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p2_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Logging the final value...
            self.p_pre = p_pre

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x2_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id7 - K @ H
            self.p2_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

            # DEBUG symmetric matrices
            # assert np.abs(self.p2_cor - self.p2_cor.T).sum() < 1e-6

        # and the smoothing
        x1_cor = np.array(self.tracked_variables["x1_cor"])
        p1_cor = np.array(self.tracked_variables["p1_cor"])
        x2_cor = np.array(self.tracked_variables["x2_cor"][::-1])
        p2_cor = np.array(self.tracked_variables["p2_cor"][::-1])

        for i in range(1, df.shape[0]):
            s1 = np.linalg.inv(p1_cor[i])
            s2 = np.linalg.inv(p2_cor[i])
            self.ps = np.linalg.inv(s1 + s2)
            self.xs = (self.ps @ (s1 @ x1_cor[i] + s2 @ x2_cor[i])).T

        filtered = pd.DataFrame(
            self.tracked_variables["xs"],
            columns=[
                "x",
                "y",
                "alt_baro",
                "alt_geo",
                "math_angle",
                "velocity",
                "vert_rate",
            ],
        )

        return data.assign(**self.postprocess(filtered))


class KalmanTaxiway(ProcessXYFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    xs: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    shl: TrackVariable[Any] = TrackVariable()

    def __init__(self) -> None:
        super().__init__()

        # bruit de mesure
        self.R = np.diag([9, 9, 2, 2, 1]) ** 2

        # plus de bruit de modèle sur les dérivées qui sont pas recalées
        self.Q = 0.5 * np.diag([0.25, 0.25, 2, 2])
        self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=2)
        self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=-2)

    def distance(
        self, prediction: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(data)

        for cumul in self.tracked_variables.values():
            cumul.clear()

        # initial state
        _id4 = np.eye(df.shape[1])
        dt = 1

        self.x_mes = df.iloc[0].values
        self.x1_cor = df.iloc[0].values
        self.p1_cor = _id4 * 1e5

        # >>> First FORWARD  <<<

        for i in range(1, df.shape[0]):
            # measurement
            self.x_mes = df.iloc[i].values

            # prediction
            A = np.eye(4) + dt * np.eye(4, k=2)
            x1_cor = self.x1_cor
            x1_cor[2:] = np.where(x1_cor[2:] == x1_cor[2:], x1_cor[2:], 0)

            x_pre = A @ x1_cor
            p_pre = A @ self.p1_cor @ A.T + self.Q

            distance, dx, dy = self.distance(x_pre)

            H = np.r_[_id4.copy(), np.array([[dx, dy, 0, 0]])]

            # DEBUG symmetric matrices
            assert np.abs(p_pre - p_pre.T).sum() < 1e-6
            assert np.all(np.linalg.eigvals(p_pre) > 0)

            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, x_pre)
            idx = np.where(self.x_mes != self.x_mes)
            H[idx, idx] = 0

            # innovation
            nu = np.r_[x_mes - x_pre, [-distance]]
            S = H @ p_pre @ H.T + self.R

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)
            # state correction
            self.x1_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id4 - K @ H
            self.p1_cor = imkh @ p_pre @ imkh.T + K @ self.R @ K.T

            # DEBUG symmetric matrices
            assert np.abs(self.p1_cor - self.p1_cor.T).sum() < 1e-6
            assert np.all(np.linalg.eigvals(self.p1_cor) > 0)

        # >>> Now BACKWARD  <<<
        self.x2_cor = self.x1_cor
        self.p2_cor = 100 * self.p1_cor
        dt = -dt

        for i in range(df.shape[0] - 1, 0, -1):
            # measurement
            self.x_mes = df.iloc[i].values

            # prediction
            A = np.eye(4) + dt * np.eye(4, k=2)
            x_pre = A @ self.x2_cor
            p_pre = A @ self.p2_cor @ A.T + self.Q

            distance, dx, dy = self.distance(x_pre)

            H = np.r_[_id4.copy(), np.array([[dx, dy, 0, 0]])]

            # DEBUG symmetric matrices
            assert np.abs(p_pre - p_pre.T).sum() < 1e-6
            assert np.all(np.linalg.eigvals(p_pre) > 0)

            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, x_pre)
            idx = np.where(self.x_mes != self.x_mes)
            H[idx, idx] = 0

            # innovation
            nu = np.r_[x_mes - x_pre, [-distance]]
            S = H @ p_pre @ H.T + self.R

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x2_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id4 - K @ H
            self.p2_cor = imkh @ p_pre @ imkh.T + K @ self.R @ K.T

            # DEBUG symmetric matrices
            assert np.abs(self.p2_cor - self.p2_cor.T).sum() < 1e-6
            assert np.all(np.linalg.eigvals(self.p2_cor) > 0)

        # and the smoothing
        x1_cor = np.array(self.tracked_variables["x1_cor"])
        p1_cor = np.array(self.tracked_variables["p1_cor"])
        x2_cor = np.array(self.tracked_variables["x2_cor"][::-1])
        p2_cor = np.array(self.tracked_variables["p2_cor"][::-1])

        for i in range(1, df.shape[0]):
            s1 = np.linalg.inv(p1_cor[i])
            s2 = np.linalg.inv(p2_cor[i])
            self.ps = np.linalg.inv(s1 + s2)
            self.xs = (self.ps @ (s1 @ x1_cor[i] + s2 @ x2_cor[i])).T

        filtered = pd.DataFrame(
            self.tracked_variables["xs"],
            columns=["x", "y", "dx", "dy"],
        )

        return data.assign(**self.postprocess(filtered))
