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

import abc
import numpy
import os
import warnings

from . import parallel
from . import support

def export(integration_results, exporters):
    """
    Exports trajectory frames written during calls to
    :py:meth:`multiopenmm.Simulation.integrate`.

    Parameters
    ----------
    integration_results : iterable of multiopenmm.IntegrationResult
        Results produced by :py:meth:`multiopenmm.Simulation.integrate`.  Each
        trajectory frame associated with each result will be processed in turn.
    exporters : iterable of multiopenmm.export.Exporter
        Exporters to extract information about particular molecular system
        instances from each frame to files.
    """

    integration_results = tuple(integration_results)
    for integration_result in integration_results:
        if not isinstance(integration_result, parallel.IntegrationResult):
            raise TypeError("integration result must be an IntegrationResult")

    exporter = tuple(exporters)
    for exporter in exporters:
        if not isinstance(exporter, Exporter):
            raise TypeError("exporter must be an Exporter")

    # Keep track of files seen so far so that we need not keep opening and
    # closing files with every new integration result.
    open_files = {}

    for integration_result in integration_results:
        path_table, indices, frame_count, path_indices, byte_offsets, particle_indices_start, particle_indices_end = integration_result._get_data()

        # Open all files not yet seen, checking that they are valid.
        for path in path_table:
            if path not in open_files:
                file = open(path, "rb")
                support.RawFileIO.read_header(file)
                open_files[path] = file

        # Keep track of data we have read so far so that we need not read it
        # more than once.
        read_data = {}

        # Create a callback for the retrieval of frame data for a particular
        # instance index.
        def get_frames(index):
            # Get the index in the integration result of the specified instance
            # index.  Return before yielding any frames if an instance index not
            # present in the integration result is specified.
            result_indices = numpy.argwhere(indices == index).flatten()
            if result_indices.size > 1:
                raise RuntimeError("invalid indices in integration result")
            if result_indices.size < 1:
                return
            result_index, = result_indices

            # Retrieve data from the integration result.
            path_index = path_indices[result_index]
            byte_offset = byte_offsets[result_index]
            particle_index_start = particle_indices_start[result_index]
            particle_index_end = particle_indices_end[result_index]

            # If we have yet to read from the indicated path at the indicated
            # offset, read and save the frames that were read.
            key = (path_index, byte_offset)
            if key not in read_data:
                file = open_files[path_table[path_index]]
                file.seek(byte_offset)
                read_data[key] = [support.RawFileIO.read_frame(file) for frame_index in range(frame_count)]

            # Apply the slice for the instance and return.
            for vectors, positions, energy in read_data[key]:
                if positions is not None:
                    positions = positions[particle_index_start:particle_index_end]
                yield vectors, positions, energy

        # Invoke each exporter.
        for exporter in exporters:
            exporter.export(get_frames)
            
    # Close all opened files.
    for file in open_files.values():
        file.close()

def delete_results(integration_results):
    """
    Deletes all temporary raw trajectory files created by calls to
    :py:meth:`multiopenmm.Simulation.integrate`.

    Parameters
    ----------
    integration_results : iterable of multiopenmm.IntegrationResult
        Results produced by :py:meth:`multiopenmm.Simulation.integrate`.  All
        temporary raw trajectory files referenced across all result objects will
        be permanently deleted.  References to files that are missing will be
        ignored.

    Notes
    -----
    This function is intended to be called after :py:func:`export` has been
    called as many times as desired to extract information from result objects
    after a run.  This function will permanently delete all of the raw
    trajectory files referenced by the result objects, even if they contain
    other trajectory frames not referenced by any of the given result objects.
    """

    integration_results = tuple(integration_results)
    for integration_result in integration_results:
        if not isinstance(integration_result, parallel.IntegrationResult):
            raise TypeError("integration result must be an IntegrationResult")

    paths = set()
    for integration_result in integration_results:
        path_table, _, _, _, _, _, _ = integration_result._get_data()
        paths.update(path_table)

    for path in sorted(paths):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

class Exporter(abc.ABC):
    """
    An abstract class that can be inherited from to define a custom exporter.
    """

    __slots__ = ()

    @abc.abstractmethod
    def export(self, get_frames):
        """
        Exports information from trajectory frames.

        Parameters
        ----------
        get_frames : callable
            A function that accepts an integer indicating the index of the
            instance for which trajectory frames are to be retrieved, and
            returns an iterable object yielding three-element tuples.  A `None`
            entry in a tuple indicates no data of this kind for a frame.  The
            first element of each tuple, if not `None`, will be a `(3, 3)`
            matrix of periodic box vector components, in row vector form.  The
            second element, if not `None`, will be a `(particle_count, 3)`
            matrix of position coordinates, and the third element, a scalar
            potential energy value.

        Notes
        -----
        This method must be implemented in derived classes.
        """

class DCDExporter(Exporter):
    """
    An exporter that writes frame vectors and positions to a DCD trajectory file
    with the specified path.

    Parameters
    ----------
    path : str
        The path to which to write.
    index : int
        The index of the instance for which to export information.
    write_vectors : bool
        Whether or not to write periodic box vector components when they are
        present.

    Notes
    -----
    `mdtraj` must be installed to create an instance of this exporter.
    """

    A_PER_NM = 10

    __slots__ = ("__mdtraj", "__path", "__index", "__write_vectors", "__file")

    def __init__(self, path, index, write_vectors=True):
        import mdtraj
        self.__mdtraj = mdtraj

        if not isinstance(path, str):
            raise TypeError("path must be a str")
        if not isinstance(index, int):
            raise TypeError("index must be an int")
        if not isinstance(write_vectors, bool):
            raise TypeError("write_vectors must be a bool")

        self.__path = str(path)
        self.__index = int(index)
        self.__write_vectors = bool(write_vectors)

        self.__file = self.__mdtraj.formats.DCDTrajectoryFile(self.__path, "w")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Closes the file to which the exporter exports.
        """

        self.__file.close()

    @property
    def path(self):
        """
        str: The path to which to write.
        """

        return self.__path

    @property
    def index(self):
        """
        int : The index of the instance for which to export information.
        """

        return self.__index

    @property
    def write_vectors(self):
        """
        bool : Whether or not to write periodic box vector components when they
        are present.
        """

        return self.__write_vectors

    def export(self, get_frames):
        """
        Exports information from trajectory frames.
        """

        for vectors, positions, _ in get_frames(self.__index):
            if positions is None:
                raise ValueError("missing positions")
            else:
                positions = positions * self.A_PER_NM

            if vectors is not None and self.__write_vectors:
                if numpy.any(numpy.triu(vectors, 1)):
                    warnings.warn("Triclinic box components not representable in DCD format", support.MultiOpenMMWarning)
                a, b, c, alpha, beta, gamma = self.__mdtraj.utils.box_vectors_to_lengths_and_angles(*vectors)
                lengths = numpy.array([a, b, c]) * self.A_PER_NM
                angles = numpy.array([alpha, beta, gamma])
            else:
                lengths = angles = None

            self.__file.write(positions, lengths, angles)

class TextEnergyExporter(Exporter):
    """
    An exporter that writes frame energies to a text file with the specified
    path, one energy per line.

    Parameters
    ----------
    path : str
        The path to which to write.
    index : int
        The index of the instance for which to export information.
    """

    __slots__ = ("__path", "__index", "__file")

    def __init__(self, path, index):
        if not isinstance(path, str):
            raise TypeError("path must be a str")
        if not isinstance(index, int):
            raise TypeError("index must be an int")

        self.__path = str(path)
        self.__index = int(index)

        self.__file = open(self.__path, "w")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Closes the file to which the exporter exports.
        """

        self.__file.close()

    @property
    def path(self):
        """
        str: The path to which to write.
        """

        return self.__path

    @property
    def index(self):
        """
        int : The index of the instance for which to export information.
        """

        return self.__index

    def export(self, get_frames):
        """
        Exports information from trajectory frames.
        """

        for _, _, energy in get_frames(self.__index):
            if energy is None:
                raise ValueError("missing energy")

            print(f"{energy:24.16e}", file=self.__file)
