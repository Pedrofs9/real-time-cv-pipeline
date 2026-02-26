from abc import ABC, abstractmethod
from typing import Any, Dict

class ProcessingPipeline(ABC):
    """
    Abstract base class for all processing pipelines.
    Timing and metrics are handled at the subclass level using Prometheus histograms,
    not stored in instance state.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    async def process(self, input_data: Any) -> Dict:
        """Main processing method — must be implemented by subclasses."""
        pass