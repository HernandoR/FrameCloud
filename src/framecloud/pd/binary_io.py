"""Binary file I/O implementation for pandas-based PointCloud."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger



class BinaryIO:
    """Mixin providing binary buffer/file I/O operations for pandas PointCloud."""

    @classmethod
    def from_binary_buffer(cls, bytes_buffer: bytes,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Load a PointCloud from a binary buffer.

        Args:
            bytes_buffer (bytes): Bytes buffer containing the binary data.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].

        Returns:
            PointCloud: The loaded PointCloud object.
        """
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        # [X, Y, Z, ...] must be in the attribute_names
        if not all(col in attribute_names for col in ["X", "Y", "Z"]):
            logger.error(f"Attribute names must include 'X', 'Y', and 'Z'.")
            raise ValueError(f"Attribute names must include 'X', 'Y', and 'Z'.")

        logger.info("Loading PointCloud from binary buffer.")
        array = np.frombuffer(bytes_buffer, dtype=dtype)
        num_attributes = len(attribute_names)
        if array.size % num_attributes != 0:
            logger.error(
                "Binary buffer size is not compatible with the number of attributes."
            )
            raise ValueError(
                "Binary buffer size is not compatible with the number of attributes."
            )
        array = array.reshape((-1, num_attributes))

        data = {name: array[:, i] for i, name in enumerate(attribute_names)}
        df = pd.DataFrame(data)
        pc = cls(data=df)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    def to_binary_buffer(
        self,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> bytes:
        """Save a PointCloud to a binary buffer.

        Args:
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].

        Returns:
            bytes: Bytes buffer containing the binary data.
        """
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info("Saving PointCloud to binary buffer.")
        arrays = []
        for name in attribute_names:
            if name in self.data.columns:
                arrays.append(self.data[name].to_numpy())
            else:
                logger.error(f"Attribute '{name}' not found in point cloud.")
                raise ValueError(f"Attribute '{name}' not found in point cloud.")

        combined_array = np.vstack(arrays).T.astype(dtype)
        bytes_buffer = combined_array.tobytes()
        logger.info("PointCloud saved to binary buffer successfully.")
        return bytes_buffer

    @classmethod
    def from_binary_file(cls, file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Load a PointCloud from a binary file.

        Args:
            file_path (Path): Path to the binary file ending with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].

        Returns:
            PointCloud: The loaded PointCloud object.
        """
        buffer = Path(file_path).read_bytes()
        return cls.from_binary_buffer(buffer, attribute_names, dtype)

    def to_binary_file(
        self,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a binary file.

        Args:
            file_path (Path): Path to the output binary file ending with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        bytes_buffer = self.to_binary_buffer(attribute_names, dtype)
        Path(file_path).write_bytes(bytes_buffer)
        logger.info(f"PointCloud saved to {file_path} successfully.")
