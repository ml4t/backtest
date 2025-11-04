# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing functions to compress, decompress, and manage pickling configurations.

!!! info
    For default settings, see `vectorbtpro._settings.pickling`.
"""

import ast
import io
import zipfile
from pathlib import Path

import humanize
import numpy as np

import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.checks import Comparable, is_deep_equal, is_hashable
from vectorbtpro.utils.formatting import Prettified, prettify_dict
from vectorbtpro.utils.path_ import check_mkdir

PickleableT = tp.TypeVar("PickleableT", bound="Pickleable")

__all__ = [
    "dumps",
    "loads",
    "save_bytes",
    "load_bytes",
    "save",
    "load",
    "RecState",
    "RecInfo",
    "Pickleable",
    "pdict",
    "get_id_from_class",
    "get_class_from_id",
]


def get_serialization_extensions(cls_name: tp.Optional[str] = None) -> tp.Set[str]:
    """Return all supported serialization extensions.

    Args:
        cls_name (Optional[str]): Class name to retrieve specific serialization extensions.

            If omitted, returns a union of all serialization extensions.

    Returns:
        Set[str]: Set of serialization file extensions.

    !!! info
        For default settings, see `vectorbtpro._settings.pickling`.
    """
    from vectorbtpro._settings import settings

    pickling_cfg = settings["pickling"]

    if cls_name is None:
        return set.union(*pickling_cfg["extensions"]["serialization"].values())
    return pickling_cfg["extensions"]["serialization"][cls_name]


def get_compression_extensions(cls_name: tp.Optional[str] = None) -> tp.Set[str]:
    """Return all supported compression extensions.

    Args:
        cls_name (Optional[str]): Class name to retrieve specific compression extensions.

            If omitted, returns a union of all compression extensions.

    Returns:
        Set[str]: Set of compression file extensions.

    !!! info
        For default settings, see `vectorbtpro._settings.pickling`.
    """
    from vectorbtpro._settings import settings

    pickling_cfg = settings["pickling"]

    if cls_name is None:
        return set.union(*pickling_cfg["extensions"]["compression"].values())
    return pickling_cfg["extensions"]["compression"][cls_name]


def compress(
    bytes_: bytes,
    compression: tp.CompressionLike = None,
    file_name: tp.Optional[str] = None,
    **compress_kwargs,
) -> bytes:
    """Compress given bytes using the specified compression format.

    Args:
        bytes_ (bytes): Byte stream to be compressed.
        compression (CompressionLike): Compression algorithm.

            If `True`, uses the default compression algorithm from settings.
            For options, see `extensions.compression` in `vectorbtpro._settings.pickling`.
        file_name (Optional[str]): Name of the file in the archive when using archive-based compression.
        **compress_kwargs: Keyword arguments for the compression function
            of the compression package.

    Returns:
        bytes: Compressed data.

    !!! info
        For default settings, see `extensions.compression` in `vectorbtpro._settings.pickling`.
    """
    from vectorbtpro.utils.module_ import (
        assert_can_import,
        assert_can_import_any,
        check_installed,
    )

    if isinstance(compression, bool) and compression:
        from vectorbtpro._settings import settings

        pickling_cfg = settings["pickling"]

        compression = pickling_cfg["compression"]
        if compression is None:
            raise ValueError("Set default compression in settings")
    if compression not in (None, False):
        if compression.lower() in get_compression_extensions("zip"):
            zip_buffer = io.BytesIO()
            if "compression" not in compress_kwargs:
                compress_kwargs["compression"] = zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(zip_buffer, "w", **compress_kwargs) as zip_file:
                if file_name is None:
                    file_name = "data.bin"
                zip_file.writestr(file_name, bytes_)
            bytes_ = zip_buffer.getvalue()
        elif compression.lower() in get_compression_extensions("bz2"):
            import bz2

            bytes_ = bz2.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("gzip"):
            import gzip

            bytes_ = gzip.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("lzma"):
            import lzma

            bytes_ = lzma.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("lz4"):
            assert_can_import("lz4")

            import lz4.frame

            bytes_ = lz4.frame.compress(bytes_, return_bytearray=True, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc1"):
            assert_can_import("blosc")

            import blosc

            bytes_ = blosc.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc2"):
            assert_can_import("blosc2")

            import blosc2

            if "_ignore_multiple_size" not in compress_kwargs:
                compress_kwargs["_ignore_multiple_size"] = True
            bytes_ = blosc2.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc"):
            assert_can_import_any("blosc2", "blosc")

            if check_installed("blosc2"):
                import blosc2

                if "_ignore_multiple_size" not in compress_kwargs:
                    compress_kwargs["_ignore_multiple_size"] = True
                bytes_ = blosc2.compress(bytes_, **compress_kwargs)
            else:
                import blosc

                bytes_ = blosc.compress(bytes_, **compress_kwargs)
        else:
            raise ValueError(f"Invalid compression format: '{compression}'")
    return bytes_


def decompress(
    bytes_: bytes,
    compression: tp.CompressionLike = None,
    file_name: tp.Optional[str] = None,
    **decompress_kwargs,
) -> bytes:
    """Decompress given bytes using the specified compression format.

    Args:
        bytes_ (bytes): Compressed byte stream to be decompressed.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        file_name (Optional[str]): Name of the file in the archive when using archive-based compression.
        **decompress_kwargs: Keyword arguments for the decompression function
            of the compression package.

    Returns:
        bytes: Decompressed data.

    !!! info
        For default settings, see `extensions.compression` in `vectorbtpro._settings.pickling`.
    """
    from vectorbtpro.utils.module_ import (
        assert_can_import,
        assert_can_import_any,
        check_installed,
    )

    if isinstance(compression, bool) and compression:
        from vectorbtpro._settings import settings

        pickling_cfg = settings["pickling"]

        compression = pickling_cfg["compression"]
        if compression is None:
            raise ValueError("Set default compression in settings")
    if compression not in (None, False):
        if compression.lower() in get_compression_extensions("zip"):
            zip_buffer = io.BytesIO(bytes_)
            with zipfile.ZipFile(zip_buffer, "r", **decompress_kwargs) as zip_file:
                namelist = zip_file.namelist()
                if len(namelist) == 0:
                    raise ValueError("ZIP archive is empty")
                if file_name is not None:
                    if file_name not in namelist:
                        raise FileNotFoundError(f"'{file_name}' not found in the ZIP archive")
                else:
                    if len(namelist) == 1:
                        file_name = namelist[0]
                    else:
                        raise ValueError(
                            "Multiple files exist in the ZIP archive. Please specify a filename."
                        )
                with zip_file.open(file_name) as file:
                    bytes_ = file.read()
        elif compression.lower() in get_compression_extensions("bz2"):
            import bz2

            bytes_ = bz2.decompress(bytes_, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("gzip"):
            import gzip

            bytes_ = gzip.decompress(bytes_, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("lzma"):
            import lzma

            bytes_ = lzma.decompress(bytes_, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("lz4"):
            assert_can_import("lz4")

            import lz4.frame

            bytes_ = lz4.frame.decompress(bytes_, return_bytearray=True, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc1"):
            assert_can_import("blosc")

            import blosc

            bytes_ = blosc.decompress(bytes_, as_bytearray=True, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc2"):
            assert_can_import("blosc2")

            import blosc2

            bytes_ = blosc2.decompress(bytes_, as_bytearray=True, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc"):
            assert_can_import_any("blosc2", "blosc")

            if check_installed("blosc2"):
                import blosc2

                bytes_ = blosc2.decompress(bytes_, as_bytearray=True, **decompress_kwargs)
            else:
                import blosc

                bytes_ = blosc.decompress(bytes_, as_bytearray=True, **decompress_kwargs)
        else:
            raise ValueError(f"Invalid compression format: '{compression}'")
    return bytes_


def dumps(
    obj: tp.Any,
    compression: tp.CompressionLike = None,
    compress_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> bytes:
    """Serialize an object to a byte stream, optionally compressing the result.

    Uses `dill` for pickling if available and otherwise falls back to the standard library `pickle`.
    Compression is applied using `compress`.

    Args:
        obj (Any): Object to serialize.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        compress_kwargs (KwargsLike): Keyword arguments for compression.
        **kwargs: Keyword arguments for the pickling library's `dumps` method.

    Returns:
        bytes: Serialized and optionally compressed byte stream.
    """
    from vectorbtpro.utils.module_ import warn_cannot_import

    if warn_cannot_import("dill"):
        import pickle
    else:
        import dill as pickle

    bytes_ = pickle.dumps(obj, **kwargs)
    if compress_kwargs is None:
        compress_kwargs = {}
    return compress(bytes_, compression=compression, **compress_kwargs)


def loads(
    bytes_: bytes,
    compression: tp.CompressionLike = None,
    decompress_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Any:
    """Deserialize an object from a byte stream, decompressing it if necessary.

    Uses `dill` for unpickling when available, otherwise falls back to the standard library `pickle`.
    Decompression is applied using `decompress`.

    Args:
        bytes_ (bytes): Byte stream containing the serialized object.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        decompress_kwargs (KwargsLike): Keyword arguments for decompression.
        **kwargs: Keyword arguments for the pickling library's `loads` method.

    Returns:
        Any: Deserialized object.
    """
    from vectorbtpro.utils.module_ import warn_cannot_import

    if warn_cannot_import("dill"):
        import pickle
    else:
        import dill as pickle

    if decompress_kwargs is None:
        decompress_kwargs = {}
    bytes_ = decompress(bytes_, compression=compression, **decompress_kwargs)
    return pickle.loads(bytes_, **kwargs)


def suggest_compression(file_name: str) -> tp.Optional[str]:
    """Suggest a compression algorithm based on the file name extension.

    Args:
        file_name (str): Name of the file.

    Returns:
        Optional[str]: Suggested compression algorithm if recognized; otherwise, None.
    """
    suffixes = [suffix.lower() for suffix in file_name.split(".")[1:]]
    if len(suffixes) > 0 and suffixes[-1] in get_compression_extensions():
        compression = suffixes[-1]
    else:
        compression = None
    return compression


def save_bytes(
    bytes_: bytes,
    path: tp.PathLike,
    mkdir_kwargs: tp.KwargsLike = None,
    compression: tp.CompressionLike = None,
    compress_kwargs: tp.KwargsLike = None,
) -> Path:
    """Write a byte stream to a file with optional compression.

    This function compresses the byte stream using `compress` if a compression algorithm is determined,
    either explicitly or based on the file's extension.

    Args:
        bytes_ (bytes): Byte stream containing the serialized object.
        path (PathLike): Destination file path.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        compress_kwargs (KwargsLike): Keyword arguments for compression.

    Returns:
        Path: Path to the written file.
    """
    path = Path(path)
    file_name = None
    if compression is None:
        compression = suggest_compression(path.name)
        if compression is not None:
            file_name = path.with_suffix("").name
    if file_name is not None:
        if compress_kwargs is None:
            compress_kwargs = {}
        if "file_name" not in compress_kwargs:
            compress_kwargs = dict(compress_kwargs)
            compress_kwargs["file_name"] = file_name
    if compress_kwargs is None:
        compress_kwargs = {}
    bytes_ = compress(bytes_, compression=compression, **compress_kwargs)
    if mkdir_kwargs is None:
        mkdir_kwargs = {}
    check_mkdir(path.parent, **mkdir_kwargs)
    with open(path, "wb") as f:
        f.write(bytes_)
    return path


def load_bytes(
    path: tp.PathLike,
    compression: tp.CompressionLike = None,
    decompress_kwargs: tp.KwargsLike = None,
) -> bytes:
    """Read a byte stream from a file with optional decompression.

    This function reads the file and applies decompression using `decompress` if a
    compression algorithm is determined, either explicitly or based on the file's extension.

    Args:
        path (PathLike): File path to read.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        decompress_kwargs (KwargsLike): Keyword arguments for decompression.

    Returns:
        bytes: Read and optionally decompressed byte stream.
    """
    path = Path(path)
    with open(path, "rb") as f:
        bytes_ = f.read()
    if compression is None:
        compression = suggest_compression(path.name)
    if decompress_kwargs is None:
        decompress_kwargs = {}
    return decompress(bytes_, compression=compression, **decompress_kwargs)


def save(
    obj: tp.Any,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
    compression: tp.CompressionLike = None,
    compress_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> Path:
    """Serialize an object and write it to a file with optional compression.

    This function serializes the object using `dumps` and writes the resulting byte stream
    to a file via `save_bytes`.

    Args:
        obj (Any): Object to serialize.
        path (Optional[PathLike]): File path where the object will be saved.

            If a directory is provided, the file name is derived from the object's class name.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        compress_kwargs (KwargsLike): Keyword arguments for compression.
        **kwargs: Keyword arguments for `dumps`.

    Returns:
        Path: Path to the saved file.
    """
    bytes_ = dumps(obj, **kwargs)
    if path is None:
        path = type(obj).__name__
    path = Path(path)
    if path.is_dir():
        path /= type(obj).__name__
    return save_bytes(
        bytes_,
        path,
        mkdir_kwargs=mkdir_kwargs,
        compression=compression,
        compress_kwargs=compress_kwargs,
    )


def load(
    path: tp.PathLike,
    compression: tp.CompressionLike = None,
    decompress_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Any:
    """Read a byte stream from a file and deserialize the contained object.

    This function uses `load_bytes` to read and decompress the byte stream,
    then deserializes the object using `loads`.

    Args:
        path (PathLike): File path from which to load the object.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        decompress_kwargs (KwargsLike): Keyword arguments for decompression.
        **kwargs: Keyword arguments for `loads`.

    Returns:
        Any: Deserialized object.
    """
    bytes_ = load_bytes(
        path,
        compression=compression,
        decompress_kwargs=decompress_kwargs,
    )
    return loads(bytes_, **kwargs)


@define
class RecState(DefineMixin):
    """Class representing the reconstruction state for an instance."""

    init_args: tp.Args = define.field(factory=tuple)
    """Positional arguments for instance initialization."""

    init_kwargs: tp.Kwargs = define.field(factory=dict)
    """Keyword arguments for instance initialization."""

    attr_dct: tp.Kwargs = define.field(factory=dict)
    """Mapping of attribute names to their writable values."""


@define
class RecInfo(DefineMixin):
    """Class for encapsulating information required to reconstruct an instance."""

    id_: str = define.field()
    """Unique reconstruction identifier."""

    cls: tp.Type = define.field()
    """Class associated with reconstruction."""

    modify_state: tp.Optional[tp.Callable[[RecState], RecState]] = define.field(default=None)
    """Optional callback that modifies the reconstruction state."""

    def register(self) -> None:
        """Register this instance in `rec_info_registry` using its identifier.

        Returns:
            None
        """
        rec_info_registry[self.id_] = self


rec_info_registry = {}
"""Registry of `RecInfo` instances keyed by their `id_`.

This registry is used during unpickling to reconstruct instances when needed.
"""


def get_id_from_class(obj: tp.Any) -> tp.Optional[str]:
    """Obtain the reconstruction identifier for a class or instance.

    If the object is an instance or subclass of `Pickleable` with a defined `_rec_id`, that value is returned.
    Otherwise, returns the fully qualified class path derived using `vectorbtpro.utils.module_.find_class`.

    Args:
        obj (Any): Class or instance to evaluate.

    Returns:
        Optional[str]: Reconstruction identifier or class path, or None if not found.
    """
    from vectorbtpro.utils.module_ import find_class

    if isinstance(obj, type):
        cls = obj
    else:
        cls = type(obj)
    if issubclass(cls, Pickleable):
        if cls._rec_id is not None:
            if not isinstance(cls._rec_id, str):
                raise TypeError(f"Reconstructing id of class {cls} must be a string")
            return cls._rec_id
    class_path = cls.__module__ + "." + cls.__name__
    if find_class(class_path) is not None:
        return class_path
    return None


def get_class_from_id(class_id: str) -> tp.Optional[tp.Type]:
    """Retrieve a class object from its reconstruction identifier.

    Args:
        class_id (str): Reconstruction identifier of the class.

    Returns:
        Type: Class associated with the provided identifier.
    """
    from vectorbtpro.utils.module_ import find_class

    if class_id in rec_info_registry:
        return rec_info_registry[class_id].cls
    cls = find_class(class_id)
    if cls is not None:
        return cls
    raise ValueError(f"Please register an instance of RecInfo for '{class_id}'")


def reconstruct(cls: tp.Union[tp.Hashable, tp.Type], rec_state: RecState) -> tp.Any:
    """Reconstruct an instance from a given class (or identifier) and reconstruction state.

    The function uses the reconstruction state to initialize a new instance, setting initialization
    arguments and updating attributes. If the provided class is not directly a type, it attempts to
    resolve the class using `rec_info_registry` or `vectorbtpro.utils.module_.find_class`.

    Args:
        cls (Union[Hashable, Type]): Class or its reconstruction identifier.
        rec_state (RecState): State used for reconstruction, including initialization
            arguments and attribute values.

    Returns:
        Any: Reconstructed instance.
    """
    from vectorbtpro.utils.module_ import find_class

    found_rec = False
    if not isinstance(cls, type):
        class_id = cls
        if class_id in rec_info_registry:
            found_rec = True
            cls = rec_info_registry[class_id].cls
            modify_state = rec_info_registry[class_id].modify_state
            if modify_state is not None:
                rec_state = modify_state(rec_state)
    if not isinstance(cls, type):
        if isinstance(cls, str):
            cls_name = cls
            cls = find_class(cls_name)
            if cls is None:
                cls = cls_name
    if not isinstance(cls, type):
        raise ValueError(f"Please register an instance of RecInfo for '{cls}'")
    if not found_rec:
        class_path = type(cls).__module__ + "." + type(cls).__name__
        if class_path in rec_info_registry:
            cls = rec_info_registry[class_path].cls
            modify_state = rec_info_registry[class_path].modify_state
            if modify_state is not None:
                rec_state = modify_state(rec_state)

    if issubclass(cls, Pickleable):
        rec_state = cls.modify_state(rec_state)
    obj = cls(*rec_state.init_args, **rec_state.init_kwargs)
    for k, v in rec_state.attr_dct.items():
        setattr(obj, k, v)
    return obj


class Pickleable(Base):
    """Class for pickle-able objects.

    If a subclass's instance cannot be pickled, override its `rec_state` property to return
    a `RecState` instance for reconstruction. If the class definition itself cannot be pickled
    (e.g., created dynamically), override its `_rec_id` with an arbitrary identifier, dump/save the class,
    and map this identifier to the class in `rec_id_map` for reconstruction.
    """

    _rec_id: tp.ClassVar[tp.Optional[str]] = None
    """Reconstruction identifier."""

    def dumps(self, rec_state_only: bool = False, **kwargs) -> bytes:
        """Serialize the instance to a pickle byte stream.

        Args:
            rec_state_only (bool): If True, serialize only the instance's reconstruction state
                for direct unpickling.
            **kwargs: Keyword arguments for `dumps`.

        Returns:
            bytes: Serialized byte stream.
        """
        if rec_state_only:
            rec_state = self.rec_state
            if rec_state is None:
                raise ValueError("Reconstruction state is None")
            return dumps(rec_state, **kwargs)
        return dumps(self, **kwargs)

    @classmethod
    def loads(
        cls: tp.Type[PickleableT], bytes_: bytes, check_type: bool = True, **kwargs
    ) -> PickleableT:
        """Reconstruct an instance from a pickle byte stream.

        If the unpickled object is an instance of `RecState`, it is transformed via `reconstruct`.

        Args:
            bytes_ (bytes): Byte stream containing the serialized object.
            check_type (bool): If True, validates that the unpickled object is an instance of the class.
            **kwargs: Keyword arguments for `loads`.

        Returns:
            Pickleable: Unpickled instance.
        """
        obj = loads(bytes_, **kwargs)
        if isinstance(obj, RecState):
            obj = reconstruct(cls, obj)
        if check_type and not isinstance(obj, cls):
            raise TypeError(f"Loaded object must be an instance of {cls}")
        return obj

    def encode_config_node(self, key: str, value: tp.Any, **kwargs) -> tp.Any:
        """Encode a configuration node.

        Args:
            key (str): Key for the configuration node.
            value (Any): Value to encode.
            **kwargs: Keyword arguments for encoding.

        Returns:
            Any: Encoded configuration node.
        """
        return value

    @classmethod
    def decode_config_node(cls, key: str, value: tp.Any, **kwargs) -> tp.Any:
        """Decode a configuration node.

        Args:
            key (str): Key for the configuration node.
            value (Any): Value to decode.
            **kwargs: Keyword arguments for decoding.

        Returns:
            Any: Decoded configuration node.
        """
        return value

    def encode_config(
        self,
        top_name: tp.Optional[str] = None,
        unpack_objects: bool = True,
        compress_unpacked: bool = True,
        use_refs: bool = True,
        use_class_ids: bool = True,
        nested: bool = True,
        to_dict: bool = False,
        parser_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Encode the instance to a configuration string based on its reconstruction state.

        This method encodes the instance in a format that can be decoded using `Pickleable.decode_config`.
        It uses the instance's `rec_state` property and raises an error if it is None. If an object cannot be
        represented as a string, it is serialized using `dumps`.

        Args:
            top_name (Optional[str]): Top-level section name.
            unpack_objects (bool): Flag to store a `Pickleable` object's reconstruction state in a separate section.

                Appends `@` and class name to the section name.
            compress_unpacked (bool): Flag to compress empty values in the reconstruction state.

                Keys in the reconstruction state will be appended with `~` to avoid collision with
                user-defined keys having the same name.
            use_refs (bool): Flag to create references for duplicate unhashable objects.

                Out of unhashable objects sharing the same id, only the first one will be defined
                while others will store the reference (`&` + key path) to the first one.
            use_class_ids (bool): Flag to substitute class objects with their identifiers.

                If `get_id_from_class` returns None, will pickle the definition.
            nested (bool): Flag indicating whether to represent sub-dictionaries as individual sections.
            to_dict (bool): Flag to treat objects as dictionaries during encoding.
            parser_kwargs (KwargsLike): Keyword arguments for `configparser.RawConfigParser`.
            **kwargs: Keyword arguments for `Pickleable.encode_config_node`.

        Returns:
            str: Encoded configuration string.

        !!! note
            The initial order of keys can be preserved only by using references.
        """
        import configparser
        from io import StringIO

        if parser_kwargs is None:
            parser_kwargs = {}
        parser = configparser.RawConfigParser(**parser_kwargs)
        parser.optionxform = str

        def _is_dict(dct, _to_dict=to_dict):
            if _to_dict:
                return isinstance(dct, dict)
            return type(dct) is dict

        def _get_path(k):
            if "@" in k:
                return k.split("@")[0].strip()
            return k

        def _is_referable(k):
            if "@" in k:
                return False
            if k.endswith("~"):
                return False
            return True

        def _preprocess_key(k):
            k = k.replace("#", "__HASH__")
            k = k.replace(":", "__COL__")
            k = k.replace("=", "__EQ__")
            return k

        # Flatten nested dicts
        if top_name is None:
            top_name = "top"
        stack = [(None, top_name, self)]
        dct = dict()
        id_paths = dict()
        id_objs = dict()
        while stack:
            parent_k, k, v = stack.pop(0)
            if not isinstance(k, str):
                raise TypeError("Dictionary keys must be strings")

            if parent_k is not None and use_refs and _is_referable(k):
                if id(v) in id_paths:
                    v = "&" + id_paths[id(v)]
                else:
                    if not is_hashable(v):
                        id_paths[id(v)] = _get_path(parent_k) + "." + _get_path(k)
                        id_objs[id(v)] = v  # keep object alive
            if _is_dict(v) and nested:
                if parent_k is not None and use_refs:
                    if parent_k is None:
                        ref_k = _get_path(k)
                    else:
                        ref_k = _get_path(parent_k) + "." + _get_path(k)
                    dct[parent_k][_get_path(k)] = "&" + ref_k
                if parent_k is None:
                    _k = k
                else:
                    _k = _get_path(parent_k) + "." + k
                dct[_k] = dict()
                if len(v) == 0:
                    v = {"_": "_"}
                i = 0
                for k2, v2 in v.items():
                    k2 = _preprocess_key(k2)
                    stack.insert(i, (_k, k2, v2))
                    i += 1
            else:
                if (unpack_objects or k == top_name) and isinstance(v, Pickleable):
                    class_id = get_id_from_class(v)
                    if class_id is None:
                        raise ValueError(f"Class {type(v)} cannot be found. Set reconstruction id.")
                    rec_state = v.rec_state
                    if rec_state is None:
                        if parent_k is None:
                            _k = _get_path(k)
                        else:
                            _k = _get_path(parent_k) + "." + _get_path(k)
                        raise ValueError(f"Must define reconstruction state for '{_k}'")
                    new_v = vars(rec_state)
                    if compress_unpacked and (
                        len(new_v["init_args"]) == 0 and len(new_v["attr_dct"]) == 0
                    ):
                        new_v = new_v["init_kwargs"]
                    else:
                        new_v = {k + "~": v for k, v in new_v.items()}
                    k = _preprocess_key(k)
                    stack.insert(0, (parent_k, k + " @" + class_id, new_v))
                else:
                    if parent_k is None:
                        dct[k] = v
                    else:
                        dct[parent_k][k] = v

        # Format config
        for k, v in dct.items():
            parser.add_section(k)
            if len(v) == 0:
                v = {"_": "_"}
            for k2, v2 in v.items():
                v2 = self.encode_config_node(k2, v2, **kwargs)
                if isinstance(v2, str):
                    if not (k2 == "_" and v2 == "_") and not v2.startswith("&"):
                        v2 = repr(v2)
                elif use_class_ids and isinstance(v2, type):
                    class_id = get_id_from_class(v2)
                    if class_id is not None:
                        v2 = "@" + class_id
                elif isinstance(v2, float) and np.isnan(v2):
                    v2 = "np.nan"
                elif isinstance(v2, float) and np.isposinf(v2):
                    v2 = "np.inf"
                elif isinstance(v2, float) and np.isneginf(v2):
                    v2 = "-np.inf"
                else:
                    try:
                        ast.literal_eval(repr(v2))
                        v2 = repr(v2)
                    except Exception:
                        try:
                            float(repr(v2))
                            v2 = repr(v2)
                        except Exception:
                            v2 = "!vbt.loads(" + repr(dumps(v2)) + ")"
                parser.set(k, k2, v2)
        with StringIO() as f:
            parser.write(f)
            str_ = f.getvalue()
        return str_

    @classmethod
    def decode_config(
        cls: tp.Type[PickleableT],
        str_: str,
        parse_literals: bool = True,
        run_code: bool = True,
        pack_objects: bool = True,
        use_refs: bool = True,
        use_class_ids: bool = True,
        code_context: tp.KwargsLike = None,
        parser_kwargs: tp.KwargsLike = None,
        check_type: bool = True,
        **kwargs,
    ) -> PickleableT:
        """Decode an instance from a configuration string.

        This function parses configuration strings and supports dot notation for nesting sections.
        It can parse configs without sections. Sections can also become sub-dictionaries if their names use
        dot notation. For example, the section `a.b` will become a sub-dictionary of the section `a`
        and the section `a.b.c` will become a sub-dictionary of the section `a.b`. You don't have to define
        the section `a` explicitly, it will automatically become the outermost key.
        Sections containing only a single pair (`_ = _`) are treated as empty dictionaries.

        Args:
            str_ (str): Configuration string to decode.
            parse_literals (bool): Detect Python literals and container types (e.g., `True`, `[]`),
                including special values like `np.nan`, `np.inf`, and `-np.inf`.
            run_code (bool): Execute Python code prefixed with `!`.

                Uses a context that includes all from `vectorbtpro.imported_star`
                along with any provided in `code_context`.
            pack_objects (bool): Instantiate and reconstruct objects specified by section class paths.

                Section names prefixed with `@` trigger the instantiation of a `RecState` object
                and reconstruction through `reconstruct`.
            use_refs (bool): Substitute reference strings prefixed with `&` with actual objects
                using a DAG constructed with `graphlib`.
            use_class_ids (bool): Replace class identifiers prefixed with `@` with corresponding classes.
            code_context (KwargsLike): Context dictionary used during execution of Python code.
            parser_kwargs (KwargsLike): Keyword arguments for `configparser.RawConfigParser`.
            check_type (bool): If True, validates that the decoded object is an instance of the class.
            **kwargs: Keyword arguments for `Pickleable.decode_config_node`.

        Returns:
            Pickleable: Decoded instance.

        !!! warning
            Unpickling byte streams and running code has important security implications. Don't attempt
            to parse configs coming from untrusted sources as those can contain malicious code!

        Examples:
            File `types.ini`:

            ```ini
            string = 'hello world'
            boolean = False
            int = 123
            float = 123.45
            exp_float = 1e-10
            nan = np.nan
            inf = np.inf
            numpy = !np.array([1, 2, 3])
            pandas = !pd.Series([1, 2, 3])
            expression = !dict(sub_dict2=dict(some="value"))
            mult_expression = !import math; math.floor(1.5)
            ```

            ```pycon
            >>> from vectorbtpro import *

            >>> vbt.pprint(vbt.pdict.load("types.ini"))
            pdict(
                string='hello world',
                boolean=False,
                int=123,
                float=123.45,
                exp_float=1e-10,
                nan=np.nan,
                inf=np.inf,
                numpy=<numpy.ndarray object at 0x7fe1bf84f690 of shape (3,)>,
                pandas=<pandas.core.series.Series object at 0x7fe1c9a997f0 of shape (3,)>,
                expression=dict(
                    sub_dict2=dict(
                        some='value'
                    )
                ),
                mult_expression=1
            )
            ```

            File `refs.ini`:

            ```ini
            [top]
            sr = &top.sr

            [top.sr @pandas.Series]
            data = [10756.12, 10876.76, 11764.33]
            index = &top.sr.index
            name = 'Open time'

            [top.sr.index @pandas.DatetimeIndex]
            data = ["2023-01-01", "2023-01-02", "2023-01-03"]
            ```

            ```pycon
            >>> vbt.pdict.load("refs.ini")["sr"]
            2023-01-01    10756.12
            2023-01-02    10876.76
            2023-01-03    11764.33
            Name: Open time, dtype: float64
            ```
        """
        import configparser
        from graphlib import TopologicalSorter

        from vectorbtpro.utils.eval_ import evaluate

        if parser_kwargs is None:
            parser_kwargs = {}
        parser = configparser.RawConfigParser(**parser_kwargs)
        parser.optionxform = str

        try:
            parser.read_string(str_)
        except configparser.MissingSectionHeaderError:
            parser.read_string("[top]\n" + str_)

        def _preprocess_key(k):
            k = k.replace("__HASH__", "#")
            k = k.replace("__COL__", ":")
            k = k.replace("__EQ__", "=")
            return k

        def _get_path(k):
            if "@" in k:
                return k.split("@")[0].strip()
            return k

        dct = {}
        has_top_section = False
        for k in parser.sections():
            k = _preprocess_key(k)
            v = dict(parser.items(k))
            if _get_path(k) == "top":
                has_top_section = True
            elif not _get_path(k).startswith("top."):
                k = "top." + k
            new_v = {}
            for k2, v2 in v.items():
                k2 = _preprocess_key(k2)
                if use_refs and v2.startswith("&") and not v2[1:].startswith("top."):
                    new_v[k2] = "&top." + v2[1:]
                else:
                    new_v[k2] = v2
            dct[k] = new_v
        if not has_top_section:
            dct = {"top": {"_": "_"}, **dct}

        def _get_class(k):
            if "@" in k:
                return k.split("@")[1].strip()
            return None

        class_map = {_get_path(k): _get_class(k) for k, v in dct.items()}
        dct = {_get_path(k): v for k, v in dct.items()}

        def _get_ref_node(ref):
            if ref in dct:
                ref_edges.add((k, (k, k2)))
                return ref
            ref_section = ".".join(ref.split(".")[:-1])
            ref_key = ref.split(".")[-1]
            if ref_section not in dct:
                raise ValueError(f"Referenced section '{ref_section}' not found")
            if ref_key not in dct[ref_section]:
                raise ValueError(f"Referenced object '{ref}' not found")
            return ref_section, ref_key

        # Parse config
        new_dct = dict()
        if code_context is None:
            code_context = {}
        else:
            code_context = dict(code_context)
        try:
            for k, v in vbt.imported_star.items():
                if k not in code_context:
                    code_context[k] = v
        except AttributeError:
            pass
        ref_edges = set()
        for k, v in dct.items():
            new_dct[k] = {}
            if len(v) == 1 and list(v.items())[0] == ("_", "_"):
                continue
            for k2, v2 in v.items():
                v2 = cls.decode_config_node(k2, v2, **kwargs)
                if isinstance(v2, str):
                    v2 = v2.strip()
                    if use_refs and v2.startswith("&"):
                        ref_node = _get_ref_node(v2[1:])
                        ref_edges.add((k, (k, k2)))
                        ref_edges.add(((k, k2), ref_node))
                    elif use_class_ids and v2.startswith("@"):
                        v2 = get_class_from_id(v2[1:])
                    elif run_code and v2.startswith("!"):
                        if v2.startswith("!vbt.loads(") and v2.endswith(")"):
                            v2 = evaluate(
                                v2[len("!vbt.") :], context={**code_context, "loads": loads}
                            )
                        else:
                            v2 = evaluate(v2.lstrip("!"), context=code_context)
                    else:
                        if parse_literals:
                            if v2 == "np.nan":
                                v2 = np.nan
                            elif v2 == "np.inf":
                                v2 = np.inf
                            elif v2 == "-np.inf":
                                v2 = -np.inf
                            else:
                                try:
                                    v2 = ast.literal_eval(v2)
                                except Exception:
                                    try:
                                        v2 = float(v2)
                                    except Exception:
                                        pass
                new_dct[k][k2] = v2
        dct = new_dct

        # Build DAG
        graph = dict()
        keys = sorted(dct.keys())
        hierarchy = [keys[0]]
        for i in range(1, len(keys)):
            while True:
                if keys[i].startswith(hierarchy[-1] + "."):
                    if hierarchy[-1] not in graph:
                        graph[hierarchy[-1]] = set()
                    graph[hierarchy[-1]].add(keys[i])
                    hierarchy.append(keys[i])
                    break
                del hierarchy[-1]
        if use_refs and len(ref_edges) > 0:
            for k1, k2 in ref_edges:
                if k1 not in graph:
                    graph[k1] = set()
                graph[k1].add(k2)
        if len(graph) > 0:
            sorter = TopologicalSorter(graph)
            topo_order = list(sorter.static_order())

            # Resolve nodes
            resolved_nodes = dict()
            for k in topo_order:
                if isinstance(k, tuple):
                    v = dct[k[0]][k[1]]
                    if use_refs and isinstance(v, str) and v.startswith("&"):
                        ref_node = _get_ref_node(v[1:])
                        v = resolved_nodes[ref_node]
                else:
                    section_dct = dict(dct[k])
                    if k in graph:
                        for k2 in graph[k]:
                            if isinstance(k2, tuple):
                                section_dct[k2[1]] = resolved_nodes[k2]
                            else:
                                _k2 = k2[len(k) + 1 :]
                                last_k = _k2.split(".")[-1]
                                d = section_dct
                                for s in _k2.split(".")[:-1]:
                                    if s not in d:
                                        d[s] = dict()
                                    d = d[s]
                                d[last_k] = resolved_nodes[k2]
                    if class_map.get(k) is not None and (pack_objects or k == "top"):
                        section_cls = class_map[k]
                        init_args = section_dct.pop("init_args~", ())
                        init_kwargs = section_dct.pop("init_kwargs~", {})
                        attr_dct = section_dct.pop("attr_dct~", {})
                        init_kwargs.update(section_dct)
                        rec_state = RecState(
                            init_args=init_args,
                            init_kwargs=init_kwargs,
                            attr_dct=attr_dct,
                        )
                        v = reconstruct(section_cls, rec_state)
                    else:
                        v = section_dct
                resolved_nodes[k] = v

            obj = resolved_nodes[topo_order[-1]]
        else:
            obj = dct["top"]
        if type(obj) is dict:
            obj = reconstruct(cls, RecState(init_kwargs=obj))
        if check_type and not isinstance(obj, cls):
            raise TypeError(f"Decoded object must be an instance of {cls}")
        return obj

    @classmethod
    def resolve_file_path(
        cls,
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        compression: tp.CompressionLike = None,
        for_save: bool = False,
    ) -> Path:
        """Resolve a file path ensuring valid file format and optional compression.

        File format and compression can be provided either via a suffix in `path`,
        or via the argument `file_format` and `compression` respectively.

        Args:
            path (Optional[PathLike]): File path, directory, or None.

                If None or a directory, the file name defaults to the class name.
            file_format (Optional[str]): Format specifier for determining the file extension.

                For options, see `extensions.serialization` in `vectorbtpro._settings.pickling`.
            compression (CompressionLike): Compression algorithm.

                See `compress`.
            for_save (bool): Resolve the file path for saving if True; otherwise, for loading.

        !!! note
            When saving, default `file_format` and `compression` values are taken from
            `vectorbtpro._settings.pickling`. When loading, the function searches for matching
            files in the current directory.

        Returns:
            Path: Resolved file path with the appropriate extensions.

        !!! info
            For default settings, see `vectorbtpro._settings.pickling`.
        """
        from vectorbtpro._settings import settings

        pickling_cfg = settings["pickling"]

        default_file_format = pickling_cfg["file_format"]
        default_compression = pickling_cfg["compression"]
        if isinstance(compression, bool) and compression:
            compression = default_compression
            if compression is None:
                raise ValueError("Set default compression in settings")

        if path is None:
            path = cls.__name__
        path = Path(path)
        if path.is_dir():
            path /= cls.__name__

        serialization_extensions = get_serialization_extensions()
        compression_extensions = get_compression_extensions()
        suffixes = [suffix[1:].lower() for suffix in path.suffixes]
        if len(suffixes) > 2:
            raise ValueError("Only two file extensions are supported: file format and compression")
        if len(suffixes) >= 1:
            if file_format is not None:
                raise ValueError("File format is already provided via file extension")
            file_format = suffixes[0]
        if len(suffixes) == 2:
            if compression is not None:
                raise ValueError("Compression is already provided via file extension")
            compression = suffixes[1]
        if file_format is not None:
            file_format = file_format.lower()
            if file_format not in serialization_extensions:
                raise ValueError(f"Invalid file format: '{file_format}'")
        if compression not in (None, False):
            compression = compression.lower()
            if compression not in compression_extensions:
                raise ValueError(f"Invalid compression format: '{compression}'")
        for _ in range(len(suffixes)):
            path = path.with_suffix("")

        if for_save:
            new_suffixes = []
            if file_format is None:
                file_format = default_file_format
            new_suffixes.append(file_format)
            if compression is None and file_format in get_serialization_extensions("pickle"):
                compression = default_compression
            if compression not in (None, False):
                if file_format not in get_serialization_extensions("pickle"):
                    raise ValueError("Compression can be used only with pickling")
                new_suffixes.append(compression)
            new_path = path.with_suffix("." + ".".join(new_suffixes))
            return new_path

        def _extensions_match(a, b, ext_type):
            from vectorbtpro._settings import settings

            pickling_cfg = settings["pickling"]

            for extensions in pickling_cfg["extensions"][ext_type].values():
                if a in extensions and b in extensions:
                    return True
            return False

        if file_format is not None:
            if compression not in (None, False):
                new_path = path.with_suffix(f".{file_format}.{compression}")
                if new_path.exists():
                    return new_path
            elif default_compression not in (None, False):
                new_path = path.with_suffix(f".{file_format}.{default_compression}")
                if new_path.exists():
                    return new_path
            else:
                new_path = path.with_suffix(f".{file_format}")
                if new_path.exists():
                    return new_path
        else:
            if compression not in (None, False):
                new_path = path.with_suffix(f".{default_file_format}.{compression}")
                if new_path.exists():
                    return new_path
            elif default_compression not in (None, False):
                new_path = path.with_suffix(f".{default_file_format}.{default_compression}")
                if new_path.exists():
                    return new_path
            else:
                new_path = path.with_suffix(f".{default_file_format}")
                if new_path.exists():
                    return new_path

        paths = []
        for p in path.parent.iterdir():
            if p.is_file():
                if p.stem.split(".")[0] == path.stem.split(".")[0]:
                    suffixes = [suffix[1:].lower() for suffix in p.suffixes]
                    if len(suffixes) == 0:
                        continue
                    if file_format is None:
                        if suffixes[0] not in serialization_extensions:
                            continue
                    else:
                        if not _extensions_match(suffixes[0], file_format, "serialization"):
                            continue
                    if compression is False:
                        if len(suffixes) >= 2:
                            continue
                    elif compression is None:
                        if len(suffixes) >= 2 and suffixes[1] not in compression_extensions:
                            continue
                    else:
                        if len(suffixes) == 1 or not _extensions_match(
                            suffixes[1], compression, "compression"
                        ):
                            continue
                    paths.append(p)
        if len(paths) == 1:
            return paths[0]
        if len(paths) > 1:
            raise ValueError(
                f"Multiple files found with path '{path}': {paths}. Please provide an extension."
            )
        error_message = f"No file found with path '{path}'"
        if file_format is not None:
            error_message += f", file format '{file_format}'"
        if compression not in (None, False):
            error_message += f", compression '{compression}'"
        raise FileNotFoundError(error_message)

    @classmethod
    def file_exists(cls, *args, **kwargs) -> bool:
        """Return whether a file exists.

        Args:
            *args: Positional arguments for `Pickleable.resolve_file_path`.
            **kwargs: Keyword arguments for `Pickleable.resolve_file_path`.
        """
        try:
            cls.resolve_file_path(*args, **kwargs)
            return True
        except FileNotFoundError:
            return False

    def save(
        self,
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        compression: tp.CompressionLike = None,
        mkdir_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        """Serialize and save the instance to a file.

        File path resolution is performed using `Pickleable.resolve_file_path`.

        Args:
            path (Optional[PathLike]): File path to save the instance.
            file_format (Optional[str]): Format specifier for determining the file extension.
            compression (CompressionLike): Compression algorithm.

                See `compress`.
            mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            **kwargs: Keyword arguments for `Pickleable.dumps` for pickle extensions
                and `Pickleable.encode_config` for config extensions.

        Returns:
            Path: File path where the instance was saved.
        """
        if mkdir_kwargs is None:
            mkdir_kwargs = {}

        path = self.resolve_file_path(
            path=path, file_format=file_format, compression=compression, for_save=True
        )
        suffixes = [suffix[1:].lower() for suffix in path.suffixes]
        if suffixes[0] in get_serialization_extensions("pickle"):
            if compression is None:
                suffixes = [suffix[1:].lower() for suffix in path.suffixes]
                if len(suffixes) > 0 and suffixes[-1] in get_compression_extensions():
                    compression = suffixes[-1]
            bytes_ = self.dumps(compression=compression, **kwargs)
            check_mkdir(path.parent, **mkdir_kwargs)
            with open(path, "wb") as f:
                f.write(bytes_)
        elif suffixes[0] in get_serialization_extensions("config"):
            str_ = self.encode_config(**kwargs)
            check_mkdir(path.parent, **mkdir_kwargs)
            with open(path, "w") as f:
                f.write(str_)
        else:
            raise ValueError(f"Invalid file extension: '{path.suffix}'")
        return path

    @classmethod
    def load(
        cls: tp.Type[PickleableT],
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        compression: tp.CompressionLike = None,
        **kwargs,
    ) -> PickleableT:
        """Deserialize and return an instance from a file.

        File path resolution is performed using `Pickleable.resolve_file_path`.

        Args:
            path (Optional[PathLike]): Path of the file to load.
            file_format (Optional[str]): Format specifier for determining the file extension.
            compression (CompressionLike): Compression algorithm.

                See `compress`.
            **kwargs: Keyword arguments for `Pickleable.loads` for pickle extensions
                and `Pickleable.decode_config` for config extensions.

        Returns:
            Pickleable: Deserialized instance.
        """
        path = cls.resolve_file_path(path=path, file_format=file_format, compression=compression)
        suffixes = [suffix[1:].lower() for suffix in path.suffixes]
        if suffixes[0] in get_serialization_extensions("pickle"):
            if compression is None:
                suffixes = [suffix[1:].lower() for suffix in path.suffixes]
                if len(suffixes) > 0 and suffixes[-1] in get_compression_extensions():
                    compression = suffixes[-1]
            with open(path, "rb") as f:
                bytes_ = f.read()
            return cls.loads(bytes_, compression=compression, **kwargs)
        elif suffixes[0] in get_serialization_extensions("config"):
            with open(path) as f:
                str_ = f.read()
            return cls.decode_config(str_, **kwargs)
        else:
            raise ValueError(f"Invalid file extension: '{path.suffix}'")

    def __sizeof__(self) -> int:
        return len(self.dumps())

    def getsize(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Return the size of this object.

        Args:
            readable (bool): Whether to use a human-readable format.
            **kwargs: Keyword arguments for `humanize.naturalsize`.

        Returns:
            Union[str, int]: Object's size as a human-readable string if `readable` is True,
                otherwise as an integer in bytes.
        """
        if readable:
            return humanize.naturalsize(self.__sizeof__(), **kwargs)
        return self.__sizeof__()

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        """Reconstruction state for recreating the object.

        Returns:
            Optional[RecState]: Reconstruction state used for object reconstruction.
        """
        return None

    @classmethod
    def modify_state(cls, rec_state: RecState) -> RecState:
        """Modify the reconstruction state prior to object reconstruction.

        Args:
            rec_state (RecState): Original reconstruction state.

        Returns:
            RecState: Modified reconstruction state.
        """
        return rec_state

    def __reduce__(self) -> tp.Union[str, tp.Tuple]:
        rec_state = self.rec_state
        if rec_state is None:
            return object.__reduce__(self)
        class_id = get_id_from_class(self)
        if class_id is None:
            cls = type(self)
        else:
            cls = class_id
        return reconstruct, (cls, rec_state)


pdictT = tp.TypeVar("pdictT", bound="pdict")


class pdict(Comparable, Pickleable, Prettified, dict):
    """Class for a pickleable dictionary that supports comparison, serialization, and prettification."""

    def load_update(
        self, path: tp.Optional[tp.PathLike] = None, clear: bool = False, **kwargs
    ) -> None:
        """Load serialized data from a file and update this dictionary instance in place.

        Args:
            path (Optional[PathLike]): File path to load data from.
            clear (bool): If True, clear the existing dictionary before updating.
            **kwargs: Keyword arguments for `pdict.load`.

        Returns:
            None
        """
        if clear:
            self.clear()
        self.update(self.load(path=path, **kwargs))

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        init_args = ()
        init_kwargs = dict(self)
        for k in list(init_kwargs):
            if not isinstance(k, str):
                if len(init_args) == 0:
                    init_args = (dict(),)
                init_args[0][k] = init_kwargs.pop(k)
        return RecState(init_args=init_args, init_kwargs=init_kwargs)

    def equals(
        self,
        other: tp.Any,
        check_types: bool = True,
        _key: tp.Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Perform a deep equality check between this dictionary and another object.

        Args:
            other (Any): Object to compare against.
            check_types (bool): Whether to verify types during comparison.
            **kwargs: Keyword arguments for `vectorbtpro.utils.checks.is_deep_equal`.

        Returns:
            bool: True if the objects are deeply equal, otherwise False.
        """
        if _key is None:
            _key = type(self).__name__
        if "only_types" in kwargs:
            del kwargs["only_types"]
        if check_types and not is_deep_equal(
            self,
            other,
            _key=_key,
            only_types=True,
            **kwargs,
        ):
            return False
        return is_deep_equal(
            dict(self),
            dict(other),
            _key=_key,
            **kwargs,
        )

    def prettify(self, **kwargs) -> str:
        return prettify_dict(self, **kwargs)

    def __repr__(self):
        return type(self).__name__ + "(" + repr(dict(self)) + ")"
