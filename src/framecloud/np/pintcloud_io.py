from pathlib import Path

import laspy
import numpy as np
import polars as pl
from loguru import logger

from framecloud.np.core import PointCloud


class PointCloudIO:
    """Class for handling input and output operations for PointCloud objects."""

    @staticmethod
    def from_las(file_path: Path | str) -> PointCloud:
        """Load a PointCloud from a LAS/LAZ file.

        Args:
            file_path (Path): Path to the LAS/LAZ file.
        Returns:
            PointCloud: The loaded PointCloud object.
        """
        logger.info(f"Loading PointCloud from LAS/LAZ file: {file_path}")
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T

        attributes = {}
        for dimension in las.point_format.dimensions:
            if dimension.name not in ["X", "Y", "Z"]:
                attributes[dimension.name] = las[dimension.name]

        pc = PointCloud(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

    @staticmethod
    def to_las(point_cloud: PointCloud, file_path: Path | str):
        """Save a PointCloud to a LAS file.

        Args:
            point_cloud (PointCloud): The PointCloud object to save.
            file_path (Path): Path to the output LAS file.
        please refer to https://laspy.readthedocs.io/en/latest/intro.html#point-format-6 and https://laspy.readthedocs.io/en/latest/intro.html#point-format-7 for supported attributes, and their names.
        """
        file_path = str(file_path)
        logger.info(f"Saving PointCloud to LAS file: {file_path}")
        header = laspy.LasHeader(point_format=7, version="1.4")
        las = laspy.LasData(header)

        las.x = point_cloud.points[:, 0]
        las.y = point_cloud.points[:, 1]
        las.z = point_cloud.points[:, 2]

        for attr_name, values in point_cloud.attributes.items():
            las[attr_name] = values

        las.write(file_path)
        logger.info(f"PointCloud saved to {file_path} successfully.")

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
        if position_cols is None:
            position_cols = ["X", "Y", "Z"]
        logger.info(f"Loading PointCloud from Parquet file: {file_path}")
        df = pl.read_parquet(file_path)
        points = df.select(position_cols).to_numpy()

        attributes = {}
        for col in df.columns:
            if col not in position_cols:
                attributes[col] = df[col].to_numpy()

        pc = PointCloud(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

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
        if position_cols is None:
            position_cols = ["X", "Y", "Z"]
        logger.info(f"Saving PointCloud to Parquet file: {file_path}")
        data = {}
        data[position_cols[0]] = point_cloud.points[:, 0]
        data[position_cols[1]] = point_cloud.points[:, 1]
        data[position_cols[2]] = point_cloud.points[:, 2]

        for attr_name, values in point_cloud.attributes.items():
            data[attr_name] = values

        df = pl.DataFrame(data)
        df.write_parquet(file_path)
        logger.info(f"PointCloud saved to {file_path} successfully.")

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
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        # [X, Y, Z, ...] must in the attribute_names
        point_attrs_pos = {}
        for i, name in enumerate(attribute_names):
            if name in ["X", "Y", "Z"]:
                point_attrs_pos[name] = i
        if len(point_attrs_pos) < 3:
            logger.error(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )
            raise ValueError(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )

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
        points = np.vstack(
            (
                array[:, point_attrs_pos["X"]],
                array[:, point_attrs_pos["Y"]],
                array[:, point_attrs_pos["Z"]],
            )
        ).T
        attributes = {}
        for i, name in enumerate(attribute_names):
            if name not in ["X", "Y", "Z"]:
                attributes[name] = array[:, i]
        pc = PointCloud(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

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
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info("Saving PointCloud to binary buffer.")
        arrays = []
        for name in attribute_names:
            if name == "X":
                arrays.append(point_cloud.points[:, 0])
            elif name == "Y":
                arrays.append(point_cloud.points[:, 1])
            elif name == "Z":
                arrays.append(point_cloud.points[:, 2])
            else:
                arrays.append(point_cloud.attributes[name])
        combined_array = np.vstack(arrays).T.astype(dtype)
        bytes_buffer = combined_array.tobytes()
        logger.info("PointCloud saved to binary buffer successfully.")
        return bytes_buffer

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
        buffer = Path(file_path).read_bytes()
        return PointCloudIO.from_binary_buffer(buffer, attribute_names, dtype)

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
        bytes_buffer = PointCloudIO.to_binary_buffer(
            point_cloud, attribute_names, dtype
        )
        Path(file_path).write_bytes(bytes_buffer)
        logger.info(f"PointCloud saved to {file_path} successfully.")

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
        array = np.load(file_path).astype(dtype)
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        # [X, Y, Z, ...] must in the attribute_names
        point_attrs_pos = {}
        for i, name in enumerate(attribute_names):
            if name in ["X", "Y", "Z"]:
                point_attrs_pos[name] = i
        if len(point_attrs_pos) < 3:
            logger.error(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )
            raise ValueError(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )

        logger.info(f"Loading PointCloud from NumPy file: {file_path}")
        points = np.vstack(
            (
                array[:, point_attrs_pos["X"]],
                array[:, point_attrs_pos["Y"]],
                array[:, point_attrs_pos["Z"]],
            )
        ).T
        attributes = {}
        for i, name in enumerate(attribute_names):
            if name not in ["X", "Y", "Z"]:
                attributes[name] = array[:, i]
        pc = PointCloud(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

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
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info(f"Saving PointCloud to NumPy file: {file_path}")
        arrays = []
        for name in attribute_names:
            if name == "X":
                arrays.append(point_cloud.points[:, 0])
            elif name == "Y":
                arrays.append(point_cloud.points[:, 1])
            elif name == "Z":
                arrays.append(point_cloud.points[:, 2])
            else:
                arrays.append(point_cloud.attributes[name])
        combined_array = np.vstack(arrays).T.astype(dtype)
        np.save(file_path, combined_array)
        logger.info(f"PointCloud saved to {file_path} successfully.")

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
        npz_data = np.load(file_path)
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]
        point_attrs_pos = {}
        for i, name in enumerate(attribute_names):
            if name in ["X", "Y", "Z"]:
                point_attrs_pos[name] = i
        if len(point_attrs_pos) < 3:
            logger.error(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )
            raise ValueError(
                f"""Attribute names must include 'X', 'Y', and 'Z', were given: {attribute_names}
                found positions: {point_attrs_pos}"""
            )
        logger.info(f"Loading PointCloud from NumPy .npz file: {file_path}")
        for name in attribute_names:
            if name not in npz_data:
                logger.error(f"Attribute '{name}' not found in .npz file.")
                raise ValueError(f"Attribute '{name}' not found in .npz file.")
        array = np.vstack([npz_data[name] for name in attribute_names]).T.astype(dtype)
        points = np.vstack(
            (
                array[:, point_attrs_pos["X"]],
                array[:, point_attrs_pos["Y"]],
                array[:, point_attrs_pos["Z"]],
            )
        ).T
        attributes = {}
        for i, name in enumerate(attribute_names):
            if name not in ["X", "Y", "Z"]:
                attributes[name] = array[:, i]
        pc = PointCloud(points=points, attributes=attributes)
        logger.info(f"Loaded PointCloud with {pc.num_points} points.")
        return pc

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
        if attribute_names is None:
            attribute_names = ["X", "Y", "Z"]

        logger.info(f"Saving PointCloud to NumPy .npz file: {file_path}")
        arrays = {}
        for name in attribute_names:
            if name == "X":
                arrays[name] = point_cloud.points[:, 0].astype(dtype)
            elif name == "Y":
                arrays[name] = point_cloud.points[:, 1].astype(dtype)
            elif name == "Z":
                arrays[name] = point_cloud.points[:, 2].astype(dtype)
            else:
                arrays[name] = point_cloud.attributes[name].astype(dtype)
        np.savez(file_path, **arrays)
        logger.info(f"PointCloud saved to {file_path} successfully.")

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
