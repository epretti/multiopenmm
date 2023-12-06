# This file is part of OpenMMREMD.
# ©2023 The Regents of the University of California.  All rights reserved.

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
import enum
import io
import numpy
import openmm
import pickle

from . import parallel

class Precision(enum.Enum):
    """
    Specifies a floating-point precision for position coordinates, velocity
    components, and periodic box vector components.
    """

    #: Single precision.
    SINGLE = enum.auto()

    #: Double precision.
    DOUBLE = enum.auto()

    @property
    def type(self):
        """
        type: A NumPy type corresponding to the specified precision.
        """

        match self:
            case self.SINGLE:
                return numpy.single
            case self.DOUBLE:
                return numpy.double
            case _:
                raise RuntimeError("unrecognized Precision")

class Thermostat(enum.Enum):
    """
    Specifies a thermostat algorithm.
    """

    #: Verlet integration with an Andersen (non-massive) thermostat.
    ANDERSEN = enum.auto()

    #: Langevin dynamics.
    LANGEVIN = enum.auto()

    #: Brownian dynamics.
    BROWNIAN = enum.auto()

class Barostat(enum.Enum):
    """
    Specifies a barostat algorithm.
    """

    #: Monte Carlo barostat with isotropic pressure and scaling.
    MC_ISOTROPIC = enum.auto()

class Simulation:
    """
    Maintains the state of an REMD simulation.

    Parameters
    ----------
    system : openmm.openmm.System
        An OpenMM system representing a single replica.
    manager : openmmremd.Manager
        A manager responsible for delegating work to OpenMM contexts running
        either in the current process or in other processes.
    ensemble : openmmremd.Ensemble
        The ensemble in which the simulation is to be conducted.
    periodic : bool, optional
        Whether or not periodic boundary conditions should be used.  By default,
        this will be determined based on whether or not any of the forces in the
        OpenMM system use periodic boundary conditions.
    precision : openmmremd.Precision, optional
        The floating-point precision used to store periodic box vector
        components, position coordinates, and velocity components.  By default,
        double precision will be used.
    seed : int, array of int, numpy.random.SeedSequence, numpy.random.BitGenerator, or numpy.random.Generator, optional
        A seed for random number generation.  If none is provided, a random seed
        will be used.  Two REMD simulations created with different seeds and
        otherwise identical parameters will produce different output.  However,
        two simulations created with identical parameters including seeds may or
        may not produce identical output, depending on whether or not the OpenMM
        platform or platforms used to run the simulations are deterministic.
        For more information, consult the
        `OpenMM user guide <http://docs.openmm.org/latest/userguide/library/04_platform_specifics.html#determinism>`_.
    """

    __slots__ = ("__system", "__manager", "__ensemble", "__precision", "__rng",
        "__periodic", "__particle_count", "__property_types", "__type",
        "__replica_count", "__stacks", "__vectors", "__positions",
        "__velocities", "__property_values")

    def __init__(self, system, manager, ensemble, periodic=None, precision=Precision.DOUBLE, seed=None):
        if not isinstance(system, openmm.System):
            raise TypeError("system must be an OpenMM System")
        if not isinstance(manager, parallel.Manager):
            raise TypeError("manager must be a Manager")
        if not isinstance(ensemble, Ensemble):
            raise TypeError("ensemble must be an Ensemble")
        if not (periodic is None or isinstance(periodic, bool)):
            raise TypeError("periodic must be a bool or None")
        if not isinstance(precision, Precision):
            raise TypeError("precision must be a Precision")

        # Make a copy of the system object to ensure that external changes to it
        # do not invalidate any of the data that we extract from it here.
        self.__system = openmm.XmlSerializer.clone(system)
        self.__manager = manager
        self.__ensemble = ensemble
        self.__precision = precision
        self.__rng = numpy.random.default_rng(seed)

        uses_periodic = self.__system.usesPeriodicBoundaryConditions()
        self.__periodic = uses_periodic if periodic is None else bool(periodic)
        if uses_periodic and not self.__periodic:
            raise ValueError("periodic must not be False if system uses periodic boundary conditions")

        self.__particle_count = self.__system.getNumParticles()
        self.__property_types = self.__ensemble.property_types
        self.__type = self.__precision.type

        # Initialize the simulation with no replicas.
        self.__replica_count = 0
        self.__stacks = numpy.zeros(0, dtype=int)
        self.__vectors = numpy.zeros((0, 3, 3), dtype=self.__type)
        self.__positions = numpy.zeros((0, self.__particle_count, 3), dtype=self.__type)
        self.__velocities = numpy.zeros((0, self.__particle_count, 3), dtype=self.__type)
        self.__property_values = {name: numpy.zeros(0, dtype=type) for name, type in self.__property_types.items()}

    @property
    def periodic(self):
        """
        bool: Whether or not periodic boundary conditions are used.
        """

        return self.__periodic

    @property
    def precision(self):
        """
        openmmremd.Precision: The floating-point precision used to store
        position coordinates, velocity components, and periodic box vector
        components.
        """

        return self.__precision

    @property
    def particle_count(self):
        """
        int: The number of particles in each replica.
        """

        return self.__particle_count


    @property
    def property_types(self):
        """
        dict: Names and dtypes of each of the ensemble-specific properties
        associated with the simulation.
        """

        return dict(self.__property_types)

    @property
    def replica_count(self):
        """
        int: The number of replicas.

        If this value is increased, new replicas will be added after the
        existing replicas.  If it is decreased, replicas at the end of the
        sequence of existing replicas will be removed.  Newly added replicas
        will have their properties set to zero.
        """

        return self.__replica_count

    @replica_count.setter
    def replica_count(self, replica_count):
        if not isinstance(replica_count, int):
            raise TypeError("replica_count must be an int")
        if replica_count < 0:
            raise ValueError("replica_count must be non-negative")

        self.__replica_count = int(replica_count)

        self.__stacks = self.__adjust_length(self.__stacks)
        self.__vectors = self.__adjust_length(self.__vectors)
        self.__positions = self.__adjust_length(self.__positions)
        self.__velocities = self.__adjust_length(self.__velocities)
        
        for name in tuple(self.__property_values.keys()):
            self.__property_values[name] = self.__adjust_length(self.__property_values[name])

    def modify_replicas(self, source_indices, destination_indices):
        """
        Reorders replicas.  The values of the properties of the destination
        replicas will be set to those of the source replicas.  This copies
        replica properties, so assignment from the same replica multiple times
        is allowed.  Assignment to the same replica multiple times is permitted,
        but if the values of the properties assigned differ between the
        assignments, their values for the destination replica are unspecified.

        Parameters
        ----------
        source_indices : int, slice, array of int, or array of bool
            Specification of replicas to retrieve.
        destination_indices : int, slice, array of int, or array of bool
            Specification of replicas to assign.
        """

        source_indices = self.__resolve_specification(source_indices)
        destination_indices = self.__resolve_specification(destination_indices)

        for array in (self.__stacks, self.__vectors, self.__positions, self.__velocities, *self.__property_values.values()):
            array[destination_indices] = array[source_indices]

    def get_stacks(self, indices=None):
        """
        Retrieves stack indices for the specified replicas.  All replicas with
        identical stack indices will be combined into single OpenMM systems
        containing non-interacting molecular systems for each replica in the
        stack.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        Returns
        -------
        array of int
            For each specified replica, its stack index.

        See also
        --------
        set_stacks
        """

        return self.__get(self.__stacks, indices)

    def get_vectors(self, indices=None):
        """
        Retrieves periodic box vectors for the specified replicas.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        Returns
        -------
        array of float
            For each specified replica, its `(3, 3)` matrix of periodic box
            vector components, in column vector form.

        See also
        --------
        set_vectors
        dump
        """

        return self.__get(self.__vectors, indices)

    def get_positions(self, indices=None):
        """
        Retrieves positions for the specified replicas.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        Returns
        -------
        array of float
            For each specified replica, its `(particle_count, 3)` position
            coordinates.

        See also
        --------
        set_positions
        dump
        """

        return self.__get(self.__positions, indices)

    def get_velocities(self, indices=None):
        """
        Retrieves velocities for the specified replicas.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        Returns
        -------
        array of float
            For each specified replica, its `(particle_count, 3)` velocity
            components.

        See also
        --------
        set_velocities
        dump
        """

        return self.__get(self.__velocities, indices)

    def get_property_values(self, name, indices=None):
        """
        Retrieves values of an ensemble-specific property for the specified
        replicas.

        Parameters
        ----------
        name : str
            The name of the property.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        Returns
        -------
        array
            For each specified replica, the value of the requested property.

        See also
        --------
        set_property_values
        """

        return self.__get(self.__property_values[name], indices)

    def set_stacks(self, stacks, indices=None):
        """
        Updates stack indices for the specified replicas.

        Parameters
        ----------
        stacks : int or array of int
            Stack indices for all or each of the specified replicas.  This
            parameter will be broadcast to an appropriate shape for the replica
            specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        See also
        --------
        get_stacks
        set_stacks_separate
        set_stacks_stacked
        """

        self.__set(self.__stacks, indices, stacks)

    def set_vectors(self, vectors, indices=None):
        """
        Updates periodic box vectors for the specified replicas.

        Parameters
        ----------
        vectors : array of float
            Periodic box vector components, in column vector form, for all or
            each of the specified replicas.  This parameter will be broadcast to
            an appropriate shape for the replica specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        See also
        --------
        get_vectors
        load
        """

        self.__set(self.__vectors, indices, vectors)

    def set_positions(self, positions, indices=None):
        """
        Updates positions for the specified replicas.

        Parameters
        ----------
        positions : array of float
            Position coordinates for all or each of the specified replicas.
            This parameter will be broadcast to an appropriate shape for the
            replica specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        See also
        --------
        get_positions
        load
        minimize
        """

        self.__set(self.__positions, indices, positions)

    def set_velocities(self, velocities, indices=None):
        """
        Updates velocities for the specified replicas.

        Parameters
        ----------
        velocities : array of float
            Velocity components for all or each of the specified replicas.  This
            parameter will be broadcast to an appropriate shape for the replica
            specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        See also
        --------
        get_velocities
        load
        maxwell_boltzmann
        """

        self.__set(self.__velocities, indices, velocities)

    def set_property_values(self, name, values, indices=None):
        """
        Updates values of an ensemble-specific property for the specified
        replicas.

        Parameters
        ----------
        name : str
            The name of the property.
        values : object or array
            The property value for all, or property values for each, of the
            specified replicas.  This parameter will be broadcast to an
            appropriate shape for the replica specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        See also
        --------
        get_property_values
        """

        self.__set(self.__property_values[name], values, indices)

    def set_stacks_separate(self):
        """
        Assigns each replica to a distinct stack, such that each replica will be
        simulated within a dedicated OpenMM system.
        
        See also
        --------
        set_stacks_stacked
        """

        self.set_stacks(numpy.arange(self.__replica_count))

    def set_stacks_stacked(self):
        """
        Assigns each replica to a single stack, such that all replicas will be
        combined into a single OpenMM system.

        See also
        --------
        set_stacks_separate
        """

        self.set_stacks(0)

    def dump(self, file, vectors=True, positions=True, velocities=True, precision=None, indices=None):
        """
        Writes periodic box vectors, positions, and velocities, as desired, for
        the specified replicas, to a file.

        Parameters
        ----------
        file : io.BufferedIOBase
            The binary file object into which to write data.
        vectors : bool, optional
            Whether or not to dump periodic box vector components to the file.
            By default, they will be dumped.
        positions : bool, optional
            Whether or not to dump position coordinates to the file.  By
            default, they will be dumped.
        velocities : bool, optional
            Whether or not to dump velocity components to the file.  By default,
            they will be dumped.
        precision : openmmremd.Precision, optional
            The floating-point precision used to dump periodic box vectors,
            positions, and velocities.  By default, the precision used for
            internal storage of these values will be used.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        See also
        --------
        load
        """

        if not isinstance(file, io.BufferedIOBase):
            raise TypeError("file must be a binary file")
        if not isinstance(vectors, bool):
            raise TypeError("vectors must be a bool")
        if not isinstance(positions, bool):
            raise TypeError("positions must be a bool")
        if not isinstance(velocities, bool):
            raise TypeError("velocities must be a bool")
        if not (precision is None or isinstance(precision, Precision)):
            raise TypeError("precision must be a Precision or None")

        target_type = self.__type if precision is None else precision.type

        # Retrieve the desired arrays, only copying them if their types need to
        # be converted.
        data = {}
        if vectors:
            data["vectors"] = self.get_vectors(indices).astype(target_type, copy=False)
        if positions:
            data["positions"] = self.get_positions(indices).astype(target_type, copy=False)
        if velocities:
            data["velocities"] = self.get_velocities(indices).astype(target_type, copy=False)

        pickle.dump(data, file)

    def load(self, file, vectors=None, positions=None, velocities=None, indices=None):
        """
        Reads periodic box vectors, positions, and velocities, as desired, for
        the specified replicas, from a file.

        Parameters
        ----------
        file : io.BufferedIOBase
            The binary file object from which to read data.
        vectors : bool, optional
            Whether or not to load periodic box vector components from the file.
            If True, and no components are found in the file, an error will be
            raised.  By default, components will be loaded if and only if they
            are found in the file.
        positions : bool, optional
            Whether or not to load position coordinates from the file.  If True,
            and no coordinates are found in the file, an error will be raised.
            By default, coordinates will be loaded if and only if they are found
            in the file.
        velocities : bool, optional
            Whether or not to load velocity components from the file.  If True,
            and no components are found in the file, an error will be raised.
            By default, components will be loaded if and only if they are found
            in the file.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.

        Notes
        -----
        If the file contains multiple blocks of data generated by
        :py:meth:`dump`, and the current file position is at the start of one
        such block, after a successful call to :py:meth:`load`, the current file
        position will either be the end of file if the block read was the final
        block in the file, or the start of the next block if not.  In other
        words, outputs of :py:meth:`dump` can be concatenated and reloaded
        sequentially.

        See also
        --------
        dump
        """

        if not isinstance(file, io.BufferedIOBase):
            raise TypeError("file must be a binary file")
        if not (vectors is None or isinstance(vectors, bool)):
            raise TypeError("vectors must be a bool or None")
        if not (positions is None or isinstance(positions, bool)):
            raise TypeError("positions must be a bool or None")
        if not (velocities is None or isinstance(velocities, bool)):
            raise TypeError("velocities must be a bool or None")

        data = pickle.load(file)

        # Update the desired arrays, only copying them if their types need to be
        # converted.  The data dictionary is allowed to handle missing keys by
        # raising a KeyError itself.
        if vectors or (vectors is None and "vectors" in data):
            self.set_vectors(data["vectors"].astype(self.__type, copy=False), indices)
        if positions or (positions is None and "positions" in data):
            self.set_positions(data["positions"].astype(self.__type, copy=False), indices)
        if velocities or (velocities is None and "velocities" in data):
            self.set_velocities(data["velocities"].astype(self.__type, copy=False), indices)
 
    def minimize(self, tolerance=None, iteration_count=None, indices=None):
        """
        Finds a local potential energy minimum for the specified replicas, and
        updates the position coordinates for those replicas.

        Parameters
        ----------
        tolerance : float
            The maximum allowable root-mean-square value of all force components
            at the potential energy minimum.  By default, the default value for
            :py:meth:`openmm.openmm.LocalEnergyMinimizer.minimize` will be used.
        iteration_count : int
            The maximum number of minimization iterations to perform, or 0 to
            proceed until the tolerance is satisfied.  By default, the default
            value for :py:meth:`openmm.openmm.LocalEnergyMinimizer.minimize`
            will be used.
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.
        """

        raise NotImplementedError

    def maxwell_boltzmann(self, indices=None):
        """
        Assigns velocity components for the specified replicas from a
        Maxwell-Boltzmann distribution.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of replicas.  By default, all replicas will be
            selected.
        """
        
        raise NotImplementedError

    def __get(self, array, indices):
        return array[self.__resolve_specification(indices)]

    def __set(self, array, indices, values):
        indices = self.__resolve_specification(indices)
        array[indices] = numpy.broadcast_to(values, (indices.size, *array.shape[1:]))

    def __resolve_specification(self, specification):
        # Converts an index, list of indices, array of indices, slice, or mask
        # into a one-dimensional list of indices suitable for indexing replicas.

        # If the specification is None, an array of shape (1, replica_count)
        # will be created and flattened to one of shape (replica_count,), thus
        # including all replicas in order in the specification as if it had been
        # slice(None).
        return numpy.atleast_1d(numpy.arange(self.__replica_count)[specification]).flatten()

    def __adjust_length(self, array):
        # Adjusts the size of the first dimension of a NumPy array to match the
        # current number of replicas, by either truncation or padding with
        # zeros.  The array, a slice of it, or a new array, might be returned.

        length = array.shape[0]

        # If no adjustment is necessary, do nothing.
        if self.__replica_count == length:
            return array

        # If the array must shrink, truncate it.
        if self.__replica_count < length:
            return array[:self.__replica_count]

        # Otherwise, the array must grow; create a new array of zeros and copy
        # in the items.
        new_array = numpy.zeros((self.__replica_count, *array.shape[1:]), dtype=array.dtype)
        new_array[:length] = array
        return new_array

class Ensemble(abc.ABC):
    """
    Represents a generic ensemble.  Subclasses represent specific ensembles.
    Adds the property ``step_length`` (:py:class:`float`) to replicas.
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def property_types(self):
        return {"step_length": float}

class CanonicalEnsemble(Ensemble):
    """
    Represents a canonical ensemble maintained by a thermostat.  Adds the
    properties ``temperature`` (:py:class:`float`) and ``thermostat_steps``
    (:py:class:`float`) to replicas.

    Parameters
    ----------
    thermostat : openmmremd.Thermostat, optional
        The thermostat algorithm to use.  By default, an Andersen thermostat
        will be used.
    """

    __slots__ = ("__thermostat",)

    def __init__(self, thermostat=Thermostat.ANDERSEN):
        super().__init__()

        if not isinstance(thermostat, Thermostat):
            raise TypeError("thermostat must be a Thermostat")

        self.__thermostat = thermostat

    @property
    def thermostat(self):
        """
        openmmremd.Thermostat: The thermostat algorithm to use.
        """

        return self.__thermostat

    @property
    def property_types(self):
        return {**super().property_types, "temperature": float, "thermostat_steps": float}

class IsothermalIsobaricEnsemble(Ensemble):
    """
    Represents an isothermal-isobaric ensemble maintained by a thermostat and a
    barostat.  Adds the properties ``temperature`` (:py:class:`float`),
    ``pressure`` (:py:class:`float`), ``thermostat_steps`` (:py:class:`float`),
    and ``barostat_steps`` (:py:class:`int`) to replicas.

    Parameters
    ----------
    thermostat : openmmremd.Thermostat, optional
        The thermostat algorithm to use.  By default, an Andersen thermostat
        will be used.
    barostat : openmmremd.Barostat, optional
        The barostat algorithm to use.  By default, an isotropic Monte Carlo
        barostat will be used.
    """

    __slots__ = ("__thermostat", "__barostat")

    def __init__(self, thermostat=Thermostat.ANDERSEN, barostat=Barostat.MC_ISOTROPIC):
        super().__init__()

        if not isinstance(thermostat, Thermostat):
            raise TypeError("thermostat must be a Thermostat")
        if not isinstance(barostat, Barostat):
            raise TypeError("barostat must be a Barostat")

        self.__thermostat = thermostat
        self.__barostat = barostat

    @property
    def thermostat(self):
        """
        openmmremd.Thermostat: The thermostat algorithm to use.
        """

        return self.__thermostat

    @property
    def barostat(self):
        """
        openmmremd.Barostat: The barostat algorithm to use.
        """

        return self.__barostat

    @property
    def property_types(self):
        return {**super().property_types, "temperature": float, "pressure": float, "thermostat_steps": float, "barostat_steps": int}
