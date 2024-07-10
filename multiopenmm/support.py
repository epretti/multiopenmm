# This file is part of MultiOpenMM.
# ©2024 The Regents of the University of California.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy
import openmm
import os
import pickle
import struct
import zlib

class MultiOpenMMError(Exception):
    """
    Raised when a MultiOpenMM-specific error condition arises.
    """

    __slots__ = ()

class MultiOpenMMWarning(Warning):
    """
    Issued when a MultiOpenMM-specific warning condition arises.
    """

    __slots__ = ()

class Arguments:
    # Holds function call arguments.

    __slots__ = ("__args", "__kwargs")

    def __init__(self, /, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def __repr__(self):
        repr_list = ", ".join([repr(arg) for arg in self.__args] + [f"{kw}={arg!r}" for kw, arg in self.__kwargs.items()])
        return f"Arguments({repr_list})"

    @property
    def args(self):
        return self.__args

    @property
    def kwargs(self):
        return dict(self.__kwargs)

    def apply_to(self, function):
        # Calls a given function with the stored arguments.

        return function(*self.__args, **self.__kwargs)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return (self.__args, self.__kwargs) == (other.__args, other.__kwargs)

def get_scratch_directory():
    """
    Retrieves the scratch directory used by MultiOpenMM for temporary files.

    Returns
    -------
    str
        The absolute path to the current scratch directory.
    """

    return os.path.realpath(_scratch_directory)

def set_scratch_directory(path=None):
    """
    Sets the scratch directory used by MultiOpenMM for temporary files.  By
    default, this will be the current working directory, unless overridden by
    setting the environment variable ``MULTIOPENMMSCRATCH``.  The directory must
    exist and be writable or errors will be raised when MultiOpenMM attempts to
    create temporary files.

    Parameters
    ----------
    path : str, optional
        A path to a directory in which MultiOpenMM should create temporary
        files.  If none is provided, the default behavior described will be
        restored.
    """

    global _scratch_directory
    
    if path is None:
        _scratch_directory = os.getenv("MULTIOPENMMSCRATCH", ".")

    else:
        if not isinstance(path, str):
            raise TypeError("path must be a str")

        _scratch_directory = str(path)

# Initialize _scratch_directory.
set_scratch_directory()

def strip_units(quantity):
    # Removes units from a quantity returned by the OpenMM Python API and
    # returns a raw scalar or array.

    return quantity.value_in_unit_system(openmm.unit.md_unit_system)

#: float: Boltzmann's constant.
K_B = strip_units(openmm.unit.BOLTZMANN_CONSTANT_kB * openmm.unit.AVOGADRO_CONSTANT_NA)

def sort_key_str_or_none(value):
    return (False, "") if value is None else (True, value)

def get_seed(rng):
    # Returns a positive random seed that will not overflow a 32-bit
    # integer.  This explicitly excludes zero, as this is sometimes interpreted
    # by OpenMM as an instruction to choose a seed non-deterministically.

    return rng.integers(1, 0x80000000)

def read_exactly(file, byte_count):
    # Reads an exactly specified number of bytes from a file, raising an error
    # if the requested number of bytes could not be read.

    result = file.read(byte_count)
    if len(result) != byte_count:
        raise MultiOpenMMError("premature end of file encountered")
    return result

class RawFileIO:
    # Handles reading and writing raw trajectory files.  The raw trajectory file
    # format is designed as a temporary format for local internal use only;
    # thus, it is not designed for portability across machine architectures or
    # installations of MultiOpenMM and may change incompatibly without notice
    # across versions of MultiOpenMM.  Do not rely on the constancy of the
    # details of the raw trajectory file format!

    # The header written at the start of raw trajectory files.
    HEADER = b"MultiOpenMM raw trajectory file\x00"

    # The type to use for writing floating-point values to raw trajectory files.
    FLOATING_TYPE = numpy.double
    
    # The number of bytes occupied by a floating-point value in a raw trajectory
    # file.
    FLOATING_BYTE_COUNT = numpy.dtype(FLOATING_TYPE).itemsize

    # The format to use for writing integer values to raw trajectory files.
    INTEGER_FORMAT = "@N"

    # The number of bytes occupied by an integer value in a raw trajectory file.
    INTEGER_BYTE_COUNT = struct.calcsize(INTEGER_FORMAT)

    @classmethod
    def write_header(cls, file):
        # Writes a header to a raw trajectory file.  The header should be
        # written once to the start of a raw trajectory file.
        
        file.write(cls.HEADER)

    @classmethod
    def read_header(cls, file):
        # Reads and checks the header from the start of a raw trajectory file.

        if file.read(len(cls.HEADER)) != cls.HEADER:
            raise MultiOpenMMError("invalid header in raw trajectory file")

    @classmethod
    def write_frame(cls, file, vectors=None, positions=None, velocities=None, potential_energy=None, kinetic_energy=None):
        # Writes a frame of data to a raw trajectory file.  vectors should be
        # None or a 3-by-3 NumPy array.  positions should be None or an N-by-3
        # NumPy array.  velocities should be None or an N-by-3 NumPy array.
        # potential_energy should be None or a scalar.  kinetic_energy should be
        # None or a scalar.

        write_vectors = vectors is not None
        write_positions = positions is not None
        write_velocities = velocities is not None
        write_potential_energy = potential_energy is not None
        write_kinetic_energy = kinetic_energy is not None

        if write_positions and write_velocities and positions.shape[0] != velocities.shape[0]:
            raise RuntimeError("invalid shapes")
        
        output = []

        frame_flags = int(write_vectors) | int(write_positions) << 1 | int(write_velocities) << 2 | int(write_potential_energy) << 3 | int(write_kinetic_energy) << 4
        output.append(bytes((frame_flags,)))

        if write_vectors:
            output.append(vectors.astype(cls.FLOATING_TYPE).tobytes())

        if write_positions or write_velocities:
            output.append(struct.pack(cls.INTEGER_FORMAT, positions.shape[0] if write_positions else velocities.shape[0]))

        if write_positions:
            output.append(positions.astype(cls.FLOATING_TYPE).tobytes())

        if write_velocities:
            output.append(velocities.astype(cls.FLOATING_TYPE).tobytes())

        if write_potential_energy:
            output.append(numpy.array(potential_energy, dtype=cls.FLOATING_TYPE).tobytes())

        if write_kinetic_energy:
            output.append(numpy.array(kinetic_energy, dtype=cls.FLOATING_TYPE).tobytes())

        file.write(b"".join(output))

    @classmethod
    def read_frame(cls, file):
        # Reads a frame of data from a raw trajectory file.

        frame_flags, = read_exactly(file, 1)
        if frame_flags >> 5:
            raise MultiOpenMMError("unknown frame flags in raw trajectory file")

        frame = {}

        if frame_flags & 1: # read_vectors
            frame["vectors"] = numpy.frombuffer(read_exactly(file, 9 * cls.FLOATING_BYTE_COUNT), dtype=cls.FLOATING_TYPE).reshape(3, 3)

        if frame_flags >> 1 & 3: # read_positions or read_velocities
            particle_count, = struct.unpack(cls.INTEGER_FORMAT, read_exactly(file, cls.INTEGER_BYTE_COUNT))

        if frame_flags >> 1 & 1: # read_positions
            frame["positions"] = numpy.frombuffer(read_exactly(file, particle_count * 3 * cls.FLOATING_BYTE_COUNT), dtype=cls.FLOATING_TYPE).reshape(particle_count, 3)

        if frame_flags >> 2 & 1: # read_velocities
            frame["velocities"] = numpy.frombuffer(read_exactly(file, particle_count * 3 * cls.FLOATING_BYTE_COUNT), dtype=cls.FLOATING_TYPE).reshape(particle_count, 3)

        if frame_flags >> 3 & 1: # read_potential_energy
            frame["potential_energy"], = numpy.frombuffer(read_exactly(file, cls.FLOATING_BYTE_COUNT), dtype=cls.FLOATING_TYPE)

        if frame_flags >> 4 & 1: # read_kinetic_energy
            frame["kinetic_energy"], = numpy.frombuffer(read_exactly(file, cls.FLOATING_BYTE_COUNT), dtype=cls.FLOATING_TYPE)

        return frame

class ResultFileIO:
    # Handles reading and writing integration result files.  The integration
    # result file format is designed as a temporary format for local internal
    # use only; thus, it is not designed for portability across machine
    # architectures or installations of MultiOpenMM and may change incompatibly
    # without notice across versions of MultiOpenMM.  Do not rely on the
    # constancy of the details of the integration result file format.
    
    # The header written at the start of integration result files.
    HEADER = b"MultiOpenMM integration results\x00"
    
    # The format to use for writing integer values to integration result files.
    INTEGER_FORMAT = "@N"

    # The number of bytes occupied by an integer value in an integration result
    # file.
    INTEGER_BYTE_COUNT = struct.calcsize(INTEGER_FORMAT)

    # The value of the level argument to provide to zlib.
    ZLIB_LEVEL = zlib.Z_BEST_COMPRESSION

    # The value of the wbits argument to provide to zlib.
    ZLIB_WBITS = -zlib.MAX_WBITS

    @classmethod
    def write_header(cls, file):
        # Writes a header to a raw trajectory file.  The header should be
        # written once to the start of a raw trajectory file.
        
        file.write(cls.HEADER)

    @classmethod
    def read_header(cls, file):
        # Reads and checks the header from the start of a raw trajectory file.

        if file.read(len(cls.HEADER)) != cls.HEADER:
            raise MultiOpenMMError("invalid header in raw trajectory file")

    @classmethod
    def write_result(cls, file, integration_result):
        # Writes an integration result to an integration result file.

        data = zlib.compress(pickle.dumps(integration_result), level=cls.ZLIB_LEVEL, wbits=cls.ZLIB_WBITS)
        file.write(struct.pack(cls.INTEGER_FORMAT, len(data)) + data)

    @classmethod
    def read_result(cls, file):
        # Reads an integration result from an integration result file.

        byte_count, = struct.unpack(cls.INTEGER_FORMAT, read_exactly(file, cls.INTEGER_BYTE_COUNT))
        return pickle.loads(zlib.decompress(read_exactly(file, byte_count), wbits=cls.ZLIB_WBITS))
