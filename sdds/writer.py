"""
Writer
------

This module contains the writing functionality of ``sdds``.
It provides a high-level function to write SDDS files in different formats, and a series of helpers.
"""
import pathlib
import struct
import sys
from typing import IO, Any, Iterable, List, Tuple, Union

import numpy as np

from sdds.classes import ENCODING, Array, Column, Data, Definition, Description, Parameter, SddsFile, get_dtype_str


def write_sdds(sdds_file: SddsFile, output_path: Union[pathlib.Path, str], mode: str = "binary", endianness: str = sys.byteorder) -> None:
    """
    Writes SddsFile object into ``output_path``.
    The byteorder will be big-endian, independent of the byteorder of the current machine.

    Args:
        sdds_file: `SddsFile` object to write
        output_path (Union[pathlib.Path, str]): `Path` object to the output SDDS file. Can be
            a `string`, in which case it will be cast to a `Path` object.
        mode (str): "binary" or "ascii"
        endianness (str): "little" or "big"
    """
    assert mode in ["binary", "ascii"], f"Invalid mode: {mode}. Must be 'binary' or 'ascii'."

    if mode == "binary" and endianness not in ["little", "big"]:
        raise ValueError(f"Invalid endianness: {endianness}. Must be 'little' or 'big'.")
    
    output_path = pathlib.Path(output_path)
    with output_path.open("wb") as outbytes:
        names = _write_header(sdds_file, outbytes, mode, None if mode == "ascii" else endianness)

        if mode == "binary":
            _write_binary_data(sdds_file, outbytes, endianness)
        elif mode == "ascii":
            _write_ascii_data(sdds_file, outbytes)


def _write_header(sdds_file: SddsFile, outbytes: IO[bytes], mode:str = "binary", endianness: str = None) -> List[str]:
    outbytes.writelines((f"{sdds_file.version}\n".encode(ENCODING),))
    if endianness is not None:
        outbytes.writelines((f"!# {endianness}-endian\n".encode(ENCODING),))
    names = []
    if sdds_file.description is not None:
        # TODO: For Description and Data, this is not implemented yet
        outbytes.write(_sdds_def_as_str(sdds_file.description).encode(ENCODING))
    for def_name in sdds_file.definitions:
        names.append(def_name)
        definition = sdds_file.definitions[def_name]
        outbytes.write(_sdds_def_as_str(definition).encode(ENCODING))
    outbytes.write(f"&data mode={mode}, &end\n".encode(ENCODING))
    return names


def _sdds_def_as_str(definition: Union[Description, Definition, Data]) -> str:
    return f"{definition.TAG} {definition.get_key_value_string()} &end\n"


###########################################################
# Binary data writing
###########################################################
def _write_binary_data(sdds_file: SddsFile, outbytes: IO[bytes], endianness: str = sys.byteorder) -> None:
    column_keys = [k for k, v in sdds_file.definitions.items() if isinstance(v, Column)]
    row_count = min([len(sdds_file.values[key]) for key in column_keys]) if len(column_keys) > 0 else 0

    # Write the number of rows
    _write_binary_numeric("long", row_count, outbytes, endianness)

    # Write the header
    _write_binary_parameters([(d,v) for d,v in sdds_file if isinstance(d, Parameter)], outbytes, endianness)
    _write_binary_arrays([(d,v) for d,v in sdds_file if isinstance(d, Array)], outbytes, endianness)

    # Write the Columns
    for i in range(row_count):
        for column in column_keys:
            defin = sdds_file.definitions[column]
            if defin.type == "string":
                _write_binary_string(sdds_file.values[column][i], outbytes, endianness)
            else:
                _write_binary_numeric(defin.type, sdds_file.values[column][i], outbytes, endianness)


def _write_binary_parameters(param_gen: Iterable[Tuple[Parameter, Any]], outbytes: IO[bytes], endianness: str = sys.byteorder):
    for param_def, value in param_gen:
        if param_def.fixed_value is None:
            if param_def.type == "string":
                _write_binary_string(value, outbytes, endianness)
            else:
                _write_binary_numeric(param_def.type, value, outbytes, endianness)


def _write_binary_arrays(array_gen: Iterable[Tuple[Array, Any]], outbytes: IO[bytes], endianness: str = sys.byteorder):
    def get_dimensions_from_array(value):
        # Return the number of items per dimension
        # For an array a[n][m], returns [n, m]
        if isinstance(value, np.ndarray) or isinstance(value, list):
            if len(value) == 0:
                return [0]
            else:
                return [len(value)] + get_dimensions_from_array(value[0])
        return []

    for array_def, value in array_gen:
        # Number of items per dimensions need to be written before the data
        elements_per_dim = get_dimensions_from_array(value)
        long_array = np.array(elements_per_dim, dtype=get_dtype_str("long", endianness=endianness))
        outbytes.write(long_array.tobytes())

        if array_def.type == "string":
            for string in value:
                _write_binary_string(string, outbytes, endianness)
        else:
            _write_binary_numeric(array_def.type, value, outbytes, endianness)


def _write_binary_string(string: str, outbytes: IO[bytes], endianness: str = sys.byteorder):
    # Write one string every time
    outbytes.write(np.array(len(string), dtype=get_dtype_str("long", endianness)).tobytes())
    outbytes.write(
        struct.pack(
            get_dtype_str("string", endianness, length=len(string)),
            string.encode(ENCODING),
        )
    )
def _write_binary_numeric(type: str, value: Union[int, float, np.ndarray], outbytes: IO[bytes], endianness: str = sys.byteorder):
    # Wirte one numeric or a numeric array
    outbytes.write(np.array(value, dtype=get_dtype_str(type, endianness=endianness)).tobytes())


###########################################################
# ASCII data writing
###########################################################
def _write_ascii_data(sdds_file: SddsFile, outbytes: IO[bytes]) -> None:
    # Write the header
    _write_ascii_parameters([(d, v) for d, v in sdds_file if isinstance(d, Parameter)], outbytes)
    _write_ascii_arrays([(d, v) for d, v in sdds_file if isinstance(d, Array)], outbytes)

    column_keys = [k for k, v in sdds_file.definitions.items() if isinstance(v, Column)]
    row_count = min([len(sdds_file.values[key]) for key in column_keys]) if len(column_keys) > 0 else 0

    # Write the number of rows
    _write_ascii_numeric("long", row_count, outbytes)
    outbytes.write("\n".encode(ENCODING))

    # Write the columns
    for i in range(row_count):
        for column in column_keys:
            defin = sdds_file.definitions[column]
            if defin.type == "string":
                _write_ascii_string(sdds_file.values[column][i], outbytes)
            else:
                _write_ascii_numeric(defin.type, sdds_file.values[column][i], outbytes)
            outbytes.write(" ".encode(ENCODING))
        outbytes.write("\n".encode(ENCODING))


def _write_ascii_parameters(param_gen: Iterable[Tuple[Parameter, Any]], outbytes: IO[bytes]):
    for param_def, value in param_gen:
        if param_def.fixed_value is None:
            if param_def.type == "string":
                _write_ascii_string(value, outbytes)
            else:
                _write_ascii_numeric(param_def.type, value, outbytes)
            outbytes.write("\n".encode(ENCODING))


def _write_ascii_arrays(array_gen: Iterable[Tuple[Array, Any]], outbytes: IO[bytes]):
    # REVIEW: is this format meaningful for array that has more than 2 dimensions
    for array_def, array_value in array_gen:
        dimensions = np.shape(array_value)
        outbytes.write(f"{' '.join(map(str, dimensions))}\n".encode(ENCODING))

        raveled_value = np.ravel(array_value)
        num_each_line = dimensions[-1]
        for i in range(0, np.prod(dimensions), num_each_line):
            for j in range(i, i+num_each_line):
                if array_def.type == "string":
                    _write_ascii_string(raveled_value[j], outbytes)
                else:
                    _write_ascii_numeric(array_def.type, raveled_value[j], outbytes)
                outbytes.write(" ".encode(ENCODING))
            outbytes.write("\n".encode(ENCODING))


def _write_ascii_string(string: str, outbytes: IO[bytes]):
    outbytes.write(f"{string}".encode(ENCODING))

def _write_ascii_numeric(type: str, value: Union[int, float, np.ndarray], outbytes: IO[bytes]):
    # f-string cannot leave a space for the positive float sign, so we need to use % instead
    # TODO: optimize the FORMATS, consider using format_string in the definition
    FORMATS = {
        "float": "%- .8f",
        "double": "%- .15e",
        "short": "%d",
        "long": "%d",
        "llong": "%d",
        "char": "%c",
        "boolean": "%d",
        "ulong64": "%d",
    }
    outbytes.write((f"{FORMATS[type]}"%value).encode(ENCODING))
