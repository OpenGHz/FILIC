import os
from abc import abstractmethod
from typing import Any, Dict, Optional
from filic.basic import ConfigurableBasis


class DataSampler(ConfigurableBasis):
    """Data sampler for sampling kinds of data.TODO: add a close method?"""

    @abstractmethod
    def compose_path(self, directory: str, round: int) -> str:
        """Compose the path to the data file. It will be called
        at starting sampling and removing. Before returning, file
        handler can be created to save data in `update` during sampling.
        Args:
            directory (str): The directory where the data will be saved.
            round (int): The round number of the data.
        Returns:
            str: The path to the data file.
        """

    def clear(self) -> None:
        """Clear the inner data buffer if any.
        Please be careful to avoid asynchronous saving
        exceptions caused by asynchronous clearing of data"""

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the data and return.
        Args:
            data (Dict[str, Any]): The data to be processed.
        Returns:
            Dict[str, Any]: The processed data, which
            will be append to the data buffer in the
            demonstration interface.
        """
        return data

    def remove(self, path: str) -> Optional[bool]:
        """Remove the data from the given or last saved path.
        If the return value is None, the demonstrate
        interface will try to remove the path."""

    def set_info(self, info: Dict[str, Any]) -> None:
        """Set the info of the data collector.
        The info is a dict that contains the information
        of the data collector, such as the name, type, etc."""
        self._info = info

    @abstractmethod
    def save(self, path: str, data: Any) -> bool:
        """Save the data to the given path.
        Args:
            path (str): The path to the data file.
            data (Any): The data to be saved. If used in
            demonstration, the data are those stored in the
            data buffer of the demonstration interface.
        Returns:
            bool: True if the data was saved successfully, False otherwise.
        """


class MockDataSampler(DataSampler):
    """Mock data sampler for testing purpose."""

    def update(self, data) -> None:
        return None

    def save(self, path: str) -> bool:
        return True

    def remove(self, path: str) -> bool:
        return True

    def compose_path(self, directory: str, round: int) -> str:
        return os.path.join(directory, f"mock_{round}.data")
