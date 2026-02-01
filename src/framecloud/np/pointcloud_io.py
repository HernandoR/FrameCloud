from pathlib import Path

import numpy as np
from loguru import logger

from framecloud.np.binary_io import BinaryIO
from framecloud.np.core import PointCloud
from framecloud.np.las_io import LasIO
from framecloud.np.numpy_io import NumpyIO
from framecloud.np.parquet_io import ParquetIO


class PointCloudIO(LasIO, ParquetIO, BinaryIO, NumpyIO):
    """Class for handling input and output operations for PointCloud objects.

    This class composes multiple I/O implementations following Protocol-based architecture.
    It inherits from specialized I/O classes that implement specific Protocol interfaces,
    providing a unified interface for all I/O operations while maintaining separation of concerns.
    """

    @staticmethod
    def from_las(file_path: Path | str) -> PointCloud:
        """Load a PointCloud from a LAS/LAZ file.

        Args:
            file_path (Path): Path to the LAS/LAZ file.
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        return LasIO.from_las(file_path)

    @staticmethod
    def to_las(point_cloud: PointCloud, file_path: Path | str):
        """Save a PointCloud to a LAS file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output LAS file.
        please refer to https://laspy.readthedocs.io/en/latest/intro.html#point-format-6 and https://laspy.readthedocs.io/en/latest/intro.html#point-format-7 for supported attributes, and their names.
        """
        return LasIO.to_las(point_cloud, file_path)

    @staticmethod
    def from_parquet(
        file_path: Path | str,
        position_cols: list[str] = None,
    ) -> PointCloud:
        """Load a PointCloud from a Parquet file.

        Args:
            file_path (Path): Path to the Parquet file.
            position_cols (list[str]): List of column names for point positions. Defaults to ["X", "Y", "Z"].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        return ParquetIO.from_parquet(file_path, position_cols)

    @staticmethod
    def to_parquet(
        point_cloud: PointCloud, file_path: Path | str, position_cols: list[str] = None
    ):
        """Save a PointCloud to a Parquet file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output Parquet file.
            position_cols (list[str]): List of column names for point positions. Defaults to ["X", "Y", "Z"].
        """
        return ParquetIO.to_parquet(point_cloud, file_path, position_cols)

    @staticmethod
    def from_binary_buffer(
        bytes_buffer: bytes,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a NumPy binary file.

        Args:
            bytes_buffer (bytes): Bytes buffer containing the NumPy binary data.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        return BinaryIO.from_binary_buffer(bytes_buffer, attribute_names, dtype)

    @staticmethod
    def to_binary_buffer(
        point_cloud: PointCloud,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> bytes:
        """Save a PointCloud to a NumPy binary buffer.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            bytes: Bytes buffer containing the NumPy binary data.
        """
        return BinaryIO.to_binary_buffer(point_cloud, attribute_names, dtype)

    @staticmethod
    def from_binary_file(
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a NumPy binary file.

        Args:
            file_path (Path): Path to the NumPy binary file end with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        return BinaryIO.from_binary_file(file_path, attribute_names, dtype)

    @staticmethod
    def to_binary_file(
        point_cloud: PointCloud,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy binary file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output NumPy binary file end with .bin.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        return BinaryIO.to_binary_file(point_cloud, file_path, attribute_names, dtype)

    @staticmethod
    def from_numpy_file(
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a NumPy .npy file.

        Args:
            file_path (Path): Path to the NumPy .npy file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        return NumpyIO.from_numpy_file(file_path, attribute_names, dtype)

    @staticmethod
    def to_numpy_file(
        point_cloud: PointCloud,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy .npy file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output NumPy .npy file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        return NumpyIO.to_numpy_file(point_cloud, file_path, attribute_names, dtype)

    @staticmethod
    def from_npz_file(
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a NumPy .npz file.

        Args:
            file_path (Path): Path to the NumPy .npz file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        return NumpyIO.from_npz_file(file_path, attribute_names, dtype)

    @staticmethod
    def to_npz_file(
        point_cloud: PointCloud,
        file_path: Path | str,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a NumPy .npz file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output NumPy .npz file.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        return NumpyIO.to_npz_file(point_cloud, file_path, attribute_names, dtype)

    @classmethod
    def from_file(
        cls,
        file_path: Path | str,
        *,
        file_type: str = None,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ) -> PointCloud:
        """Load a PointCloud from a file.
        routes to specific loader based on file extension or file_type.
        Args:
            file_path (Path): Path to the file.
            file_type (str): Type of the file. If None, inferred from file extension.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        if file_type is None:
            file_type = Path(file_path).suffix.lower()
        if file_type in [".las", ".laz"]:
            return cls.from_las(file_path)
        elif file_type == ".parquet":
            return cls.from_parquet(file_path)
        elif file_type == ".bin":
            return cls.from_binary_file(file_path, attribute_names, dtype)
        elif file_type == ".npy":
            return cls.from_numpy_file(file_path, attribute_names, dtype)
        elif file_type == ".npz":
            return cls.from_npz_file(file_path, attribute_names, dtype)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

    @classmethod
    def to_file(
        cls,
        point_cloud: PointCloud,
        file_path: Path | str,
        *,
        file_type: str = None,
        attribute_names: list[str] = None,
        dtype=np.float32,
    ):
        """Save a PointCloud to a file.
        routes to specific saver based on file extension or file_type.
        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output file.
            file_type (str): Type of the file. If None, inferred from file extension.
            attribute_names (list[str]): List of attribute names in order. Defaults to [X,Y,Z].
        """
        if file_type is None:
            file_type = Path(file_path).suffix.lower()
        if file_type in [".las", ".laz"]:
            cls.to_las(point_cloud, file_path)
        elif file_type == ".parquet":
            cls.to_parquet(point_cloud, file_path)
        elif file_type == ".bin":
            cls.to_binary_file(point_cloud, file_path, attribute_names, dtype)
        elif file_type == ".npy":
            cls.to_numpy_file(point_cloud, file_path, attribute_names, dtype)
        elif file_type == ".npz":
            cls.to_npz_file(point_cloud, file_path, attribute_names, dtype)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")
