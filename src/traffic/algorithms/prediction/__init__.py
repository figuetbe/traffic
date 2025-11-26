from typing import Protocol, runtime_checkable, Union

from ...core.flight import Flight
from ...core.traffic import Traffic


@runtime_checkable
class PredictBase(Protocol):
    def predict(self, flight: Flight) -> Union[Flight, Traffic]: ...

from .cfm import CFMPredict
from .straightline import StraightLinePredict