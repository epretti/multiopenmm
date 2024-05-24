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
import enum
import gzip
import io
import numpy
import openmm
import pickle
import scipy
import tempfile
import warnings

from . import concurrency
from . import simulation
from . import stacking
from . import support

#: float: The default periodic box vector length.
DEFAULT_VECTOR_LENGTH = 2.0

#: float: The default energy minimization tolerance.
DEFAULT_MINIMIZE_TOLERANCE = 10.0

#: int: The default energy minimization iteration count.
DEFAULT_MINIMIZE_ITERATION_COUNT = 0

#: float: The default step length property value.
DEFAULT_STEP_LENGTH = 0.002

#: float: The default constraint tolerance property value.
DEFAULT_CONSTRAINT_TOLERANCE = 0.00001

#: float: The default temperature property value.
DEFAULT_TEMPERATURE = 300.0

#: float: The default pressure property value.
DEFAULT_PRESSURE = 1.0

#: float: The default thermostat characteristic time property value.
DEFAULT_THERMOSTAT_STEPS = 100.0

#: int: The default barostat characteristic time property value.
DEFAULT_BAROSTAT_STEPS = 25

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
    Maintains the state of a parallel simulation.

    Parameters
    ----------
    templates : iterable of openmm.openmm.System
        OpenMM systems serving as templates for molecular system instances.
    manager : multiopenmm.Manager
        A manager responsible for delegating work to OpenMM contexts.
    ensemble : multiopenmm.Ensemble
        The ensemble in which the simulation is to be conducted.
    precision : multiopenmm.Precision, optional
        The floating-point precision used to store periodic box vector
        components, position coordinates, and velocity components.  By default,
        double precision will be used.
    seed : int, array of int, numpy.random.SeedSequence, numpy.random.BitGenerator, or numpy.random.Generator, optional
        A seed for random number generation.  If none is provided, a random seed
        will be used.  Two simulations created with different seeds and
        otherwise identical parameters will produce different output.  However,
        two simulations created with identical parameters including seeds may or
        may not produce identical output, depending on whether or not the OpenMM
        platform or platforms used to run the simulations, and the manager used,
        are deterministic.  For more information, consult the
        `OpenMM user guide <http://docs.openmm.org/latest/userguide/library/04_platform_specifics.html#determinism>`_.
    """

    __slots__ = ("__templates", "__manager", "__ensemble", "__precision",
        "__rng", "__template_count", "__template_sizes", "__property_types",
        "__property_defaults", "__type", "__instance_count",
        "__template_indices", "__instance_sizes", "__instance_offsets",
        "__stacks", "__vectors", "__positions", "__velocities",
        "__property_values", "__cache_directory", "__cache_index")

    def __init__(self, templates, manager, ensemble, precision=Precision.DOUBLE, seed=None):
        # Make copies of the system objects to ensure that external changes to
        # them do not invalidate any of the data that we extract from them here.
        self.__templates = []
        for template in templates:
            if not isinstance(template, openmm.System):
                raise TypeError("template must be an OpenMM System")
            self.__templates.append(openmm.XmlSerializer.clone(template))

        if not isinstance(manager, concurrency.Manager):
            raise TypeError("manager must be a Manager")
        if not isinstance(ensemble, Ensemble):
            raise TypeError("ensemble must be an Ensemble")
        if not isinstance(precision, Precision):
            raise TypeError("precision must be a Precision")

        self.__manager = manager
        self.__ensemble = ensemble
        self.__precision = Precision(precision)

        self.__rng = numpy.random.default_rng(seed)

        self.__template_count = len(self.__templates)
        self.__template_sizes = numpy.array([template.getNumParticles() for template in self.__templates], dtype=int)
        self.__property_types = self.__ensemble.property_types
        self.__property_defaults = self.__ensemble.property_defaults
        self.__type = self.__precision.type

        # Initialize the simulation with no instances.
        self.__instance_count = 0
        self.__template_indices = numpy.zeros(0, dtype=int)
        self.__instance_sizes = numpy.zeros(0, dtype=int)
        self.__instance_offsets = numpy.zeros(1, dtype=int)
        self.__stacks = numpy.zeros(0, dtype=int)
        self.__vectors = numpy.zeros((0, 3, 3), dtype=self.__type)
        self.__positions = numpy.zeros((0, 3), dtype=self.__type)
        self.__velocities = numpy.zeros((0, 3), dtype=self.__type)
        self.__property_values = {name: numpy.zeros(0, dtype=type) for name, type in self.__property_types.items()}

        # Cache combined systems created from these templates.
        self.__cache_directory = tempfile.TemporaryDirectory(prefix="multiopenmm_cache_", dir=support.get_scratch_directory(), ignore_cleanup_errors=True)
        self.__cache_index = {}

    @property
    def precision(self):
        """
        multiopenmm.Precision: The floating-point precision used to store
        position coordinates, velocity components, and periodic box vector
        components.
        """

        return self.__precision

    @property
    def template_count(self):
        """
        int: The number of template OpenMM systems.
        """

        return self.__template_count

    @property
    def property_types(self):
        """
        dict: Names and dtypes of each of the ensemble-specific properties
        associated with the simulation.
        """

        return dict(self.__property_types)

    @property
    def property_defaults(self):
        """
        dict: Names and default values for ensemble-specific properties
        associated with the simulation.  Not every property may have a default
        value, in which case its name will only appear as a key in
        :py:attr:`property_types` and not in :py:attr:`property_defaults`.  In
        this case, the property will behave as if its default value is zero.
        """

        return dict(self.__property_defaults)

    @property
    def instance_count(self):
        """
        int: The number of molecular system instances.

        If this value is increased, new instances will be added after the
        existing instances.  If it is decreased, instances at the end of the
        sequence of existing instances will be removed.  Newly added instances
        will have their properties set to default values, or zero if no default
        values exist for the properties.
        """

        return self.__instance_count

    @instance_count.setter
    def instance_count(self, instance_count):
        if not isinstance(instance_count, int):
            raise TypeError("instance_count must be an int")
        if instance_count < 0:
            raise ValueError("instance_count must be non-negative")

        self.__instance_count = int(instance_count)

        self.__template_indices = self.__adjust_per_instance(self.__template_indices)
        old_offsets = self.__update_instance_data()

        self.__stacks = self.__adjust_per_instance(self.__stacks)
        self.__vectors = self.__adjust_per_instance(self.__vectors, DEFAULT_VECTOR_LENGTH * numpy.eye(3))
        self.__positions = self.__adjust_per_particle(self.__positions, old_offsets)
        self.__velocities = self.__adjust_per_particle(self.__velocities, old_offsets)
        
        for name in tuple(self.__property_values.keys()):
            self.__property_values[name] = self.__adjust_per_instance(self.__property_values[name], self.__property_defaults.get(name, 0))

    def reorder_instances(self, source_indices, destination_indices):
        """
        Reorders instances.  The values of the properties of the destination
        instances will be set to those of the source instances.  This copies
        instance properties, so assignment from the same instance multiple times
        is allowed.  Assignment to the same instance multiple times is
        permitted, but if the values of the properties assigned differ between
        the assignments, their values for the destination instance are
        unspecified: they may come from any one of the source instances
        (although this behavior will be deterministic).

        Parameters
        ----------
        source_indices : int, slice, array of int, or array of bool
            Specification of instances to retrieve.
        destination_indices : int, slice, array of int, or array of bool
            Specification of instances to assign.

        Notes
        -----
        Currently, if the index specifications are used to index ascending
        arrays of integers starting from 0 and of the same length as the number
        of instances, and the resulting source index array is broadcasted to the
        shape of the destination index array, the property value assignments
        will take place as though values were retrieved in order using the
        source index array and set in order using the destination index array.
        This includes the case of assignment to the same instance multiple times
        (although this behavior should be considered an implementation detail).
        """

        source_indices = self.__resolve_with_size(self.__instance_count, source_indices)
        destination_indices = self.__resolve_with_size(self.__instance_count, destination_indices)
        source_indices = numpy.broadcast_to(source_indices, destination_indices.shape)

        self.__template_indices[destination_indices] = self.__template_indices[source_indices]
        old_offsets = self.__update_instance_data()

        # Determine the particle indices necessary to reorder per-particle
        # properties in accordance with the desired reordering of instances.
        resolve_indices = numpy.arange(self.__instance_count)
        resolve_indices[destination_indices] = source_indices
        particle_indices = self.__resolve_with_offsets(old_offsets, resolve_indices)

        self.__stacks[destination_indices] = self.__stacks[source_indices]
        self.__vectors[destination_indices] = self.__vectors[source_indices]
        self.__positions = self.__positions[particle_indices]
        self.__velocities = self.__velocities[particle_indices]

        for name, property_values in self.__property_values.items():
            property_values[destination_indices] = property_values[source_indices]

    def get_template_sizes(self, indices=None):
        """
        Retrieves the numbers of particles in the specified templates.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of templates.  By default, all templates will be
            selected.

        Returns
        -------
        array of int
            For each specified template OpenMM system, the number of particles
            in the system.
        """
        
        return self.__template_sizes[self.__resolve_with_size(self.__template_count, indices)]

    def get_instance_sizes(self, indices=None):
        """
        Retrieves the numbers of particles in the specified instances.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Returns
        -------
        array of int
            For each specified instance, the number of particles in its template
            OpenMM system.
        """

        return self.__get_per_instance(self.__instance_sizes, indices)

    def get_template_indices(self, indices=None):
        """
        Retrieves template indices for the specified instances.  The template
        index of each instance specifies which template OpenMM system it should
        be an instance of.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Returns
        -------
        array of int
            For each specified instance, its template index.

        See also
        --------
        set_template_indices
        """

        return self.__get_per_instance(self.__template_indices, indices)

    def get_stacks(self, indices=None):
        """
        Retrieves stack identifiers for the specified instances.  All instances
        with identical stack identifiers will be combined into single OpenMM
        systems containing non-interacting molecular systems for each instance
        in the stack.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Returns
        -------
        array of int
            For each specified instance, its stack identifier.

        See also
        --------
        set_stacks
        """

        return self.__get_per_instance(self.__stacks, indices)

    def get_vectors(self, indices=None):
        """
        Retrieves periodic box vectors for the specified instances.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Returns
        -------
        array of float
            For each specified instance, its ``(3, 3)`` matrix of periodic box
            vector components, in row vector form.

        See also
        --------
        set_vectors
        dump
        """

        return self.__get_per_instance(self.__vectors, indices)

    def get_positions(self, indices=None):
        """
        Retrieves positions for the specified instances.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Returns
        -------
        array of float
            For each specified instance, its ``(instance_size, 3)`` position
            coordinates.  The arrays of coordinates for all instances will be
            concatenated.

        See also
        --------
        set_positions
        dump
        """

        return self.__get_per_particle(self.__positions, indices)

    def get_velocities(self, indices=None):
        """
        Retrieves velocities for the specified instances.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Returns
        -------
        array of float
            For each specified instance, its ``(instance_size, 3)`` velocity
            components.  The arrays of components for all instances will be
            concatenated.

        See also
        --------
        set_velocities
        dump
        """

        return self.__get_per_particle(self.__velocities, indices)

    def get_property_values(self, name, indices=None):
        """
        Retrieves values of an ensemble-specific property for the specified
        instances.

        Parameters
        ----------
        name : str
            The name of the property.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Returns
        -------
        array
            For each specified instance, the value of the requested property.

        See also
        --------
        set_property_values
        """

        return self.__get_per_instance(self.__property_values[name], indices)

    def set_template_indices(self, template_indices, indices=None):
        """
        Updates template indices for the specified instances.

        Parameters
        ----------
        template_indices : int or array of int
            Template indices for all or each of the specified instances.  This
            parameter will be broadcast to an appropriate shape for the instance
            specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Notes
        -----
        Arrays of position vectors and velocity coordinates for instances may be
        truncated or padded with zeros as needed if the numbers of particles in
        the templates assigned to the instances change.

        See also
        --------
        get_template_indices
        """

        self.__set_per_instance(self.__template_indices, indices, template_indices)
        old_offsets = self.__update_instance_data()

        self.__positions = self.__adjust_per_particle(self.__positions, old_offsets)
        self.__velocities = self.__adjust_per_particle(self.__velocities, old_offsets)

    def set_stacks(self, stacks, indices=None):
        """
        Updates stack identifiers for the specified instances.

        Parameters
        ----------
        stacks : int or array of int
            Stack identifiers for all or each of the specified instances.  This
            parameter will be broadcast to an appropriate shape for the instance
            specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        See also
        --------
        get_stacks
        set_stacks_separate
        set_stacks_stacked
        """

        self.__set_per_instance(self.__stacks, indices, stacks)

    def set_vectors(self, vectors, indices=None):
        """
        Updates periodic box vectors for the specified instances.

        Parameters
        ----------
        vectors : array of float
            Periodic box vector components, in row vector form, for all or each
            of the specified instances.  This parameter will be broadcast to an
            appropriate shape for the instance specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        See also
        --------
        get_vectors
        load
        """

        self.__set_per_instance(self.__vectors, indices, vectors)

    def set_positions(self, positions, indices=None):
        """
        Updates positions for the specified instances.

        Parameters
        ----------
        positions : array of float
            Position coordinates for all or each of the specified instances.
            This parameter will be broadcast to an appropriate shape for the
            instance specification.  If multiple coordinates are given for
            multiple instances, the arrays of coordinates should be
            concatenated.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        See also
        --------
        get_positions
        load
        """

        self.__set_per_particle(self.__positions, indices, positions)

    def set_velocities(self, velocities, indices=None):
        """
        Updates velocities for the specified instances.

        Parameters
        ----------
        velocities : array of float
            Velocity components for all or each of the specified instances.
            This parameter will be broadcast to an appropriate shape for the
            instance specification.  If multiple components are given for
            multiple instances, the arrays of components should be concatenated.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        See also
        --------
        get_velocities
        load
        """

        self.__set_per_particle(self.__velocities, indices, velocities)

    def set_property_values(self, name, values, indices=None):
        """
        Updates values of an ensemble-specific property for the specified
        instances.

        Parameters
        ----------
        name : str
            The name of the property.
        values : object or array
            The property value for all, or property values for each, of the
            specified instances.  This parameter will be broadcast to an
            appropriate shape for the instance specification.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        See also
        --------
        get_property_values
        """

        self.__set_per_instance(self.__property_values[name], indices, values)

    def set_stacks_separate(self):
        """
        Assigns each instance to a distinct stack, such that each instance will
        be simulated within a dedicated OpenMM system.
        
        See also
        --------
        set_stacks_stacked
        """

        self.set_stacks(numpy.arange(self.__instance_count))

    def set_stacks_stacked(self):
        """
        Assigns each instance to a single stack, such that all instances will be
        combined into a single OpenMM system.

        See also
        --------
        set_stacks_separate
        """

        self.set_stacks(0)

    def dump(self, file, vectors=True, positions=True, velocities=True, precision=None, indices=None):
        """
        Writes periodic box vectors, positions, and velocities, as desired, for
        the specified instances, to a file.

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
        precision : multiopenmm.Precision, optional
            The floating-point precision used to dump periodic box vectors,
            positions, and velocities.  By default, the precision used for
            internal storage of these values will be used.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
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
        the specified instances, from a file.

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
            Specification of instances.  By default, all instances will be
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

    def apply_constraints(self, positions=True, velocities=True, indices=None):
        """
        Applies constraints to either the positions or the velocities, or both,
        of the specified instances.

        Parameters
        ----------
        positions : bool, optional
            Whether or not to apply constraints to position coordinates (by
            default, they will be applied).
        velocities : bool, optional
            Whether or not to apply constraints to velocity components (by
            default, they will be applied).
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.
        """

        if not isinstance(positions, bool):
            raise TypeError("positions must be a bool")
        if not isinstance(velocities, bool):
            raise TypeError("velocities must be a bool")

        positions = bool(positions)
        velocities = bool(velocities)

        indices = numpy.unique(self.__resolve_with_size(self.__instance_count, indices))
        combined_systems = self.__get_combined_systems(indices)

        for response, system_data in zip(self.__manager._distribute(*(
            support.Arguments(
                system_data["system_path"],
                system_data["context_data"],
                self.__rng.spawn(1)[0],
                system_data["vectors"],
                self.get_positions(system_data["indices"]) if positions or velocities else None,
                self.get_velocities(system_data["indices"]) if velocities else None,
                False,
                positions,
                velocities,
                False,
                False,
                system_data["enforce_periodic"],
                system_data["center_coordinates"],
                (simulation.Command.APPLY_CONSTRAINTS, support.Arguments(positions, velocities)),
            )
            for stack, system_data in combined_systems.items())), combined_systems.values()):

            _, positions_velocities_out = response()
            if positions:
                self.set_positions(positions_velocities_out[0], system_data["indices"])
            if velocities:
                self.set_velocities(positions_velocities_out[1 if positions else 0], system_data["indices"])

    def minimize(self, tolerance=None, iteration_count=None, indices=None):
        """
        Finds a local potential energy minimum for the specified instances, and
        updates the position coordinates for those instances.

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
            Specification of instances.  By default, all instances will be
            selected.
        """

        if tolerance is None:
            tolerance = DEFAULT_MINIMIZE_TOLERANCE
        else:
            if not isinstance(tolerance, (int, float)):
                raise TypeError("tolerance must be an int, float or None")
            tolerance = float(tolerance)

        if iteration_count is None:
            iteration_count = DEFAULT_MINIMIZE_ITERATION_COUNT
        else:
            if not isinstance(iteration_count, int):
                raise TypeError("iteration_count must be an int or None")
            iteration_count = int(iteration_count)
        
        indices = numpy.unique(self.__resolve_with_size(self.__instance_count, indices))
        combined_systems = self.__get_combined_systems(indices)

        for response, system_data in zip(self.__manager._distribute(*(
            support.Arguments(
                system_data["system_path"],
                system_data["context_data"],
                self.__rng.spawn(1)[0],
                system_data["vectors"],
                self.get_positions(system_data["indices"]),
                None,
                False,
                True,
                False,
                False,
                False,
                system_data["enforce_periodic"],
                system_data["center_coordinates"],
                (simulation.Command.MINIMIZE, support.Arguments(tolerance, iteration_count)),
            )
            for stack, system_data in combined_systems.items())), combined_systems.values()):

            _, (positions_out,) = response()
            self.set_positions(positions_out, system_data["indices"])

    def maxwell_boltzmann(self, indices=None):
        """
        Assigns velocity components for the specified instances from a
        Maxwell-Boltzmann distribution.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.
        """

        indices = numpy.unique(self.__resolve_with_size(self.__instance_count, indices))
        combined_systems = self.__get_combined_systems(indices)

        for response, system_data in zip(self.__manager._distribute(*(
            support.Arguments(
                system_data["system_path"],
                system_data["context_data"],
                self.__rng.spawn(1)[0],
                system_data["vectors"],
                self.get_positions(system_data["indices"]),
                None,
                False,
                False,
                True,
                False,
                False,
                system_data["enforce_periodic"],
                system_data["center_coordinates"],
                (simulation.Command.MAXWELL_BOLTZMANN, support.Arguments(system_data["run_temperature"])),
            )
            for stack, system_data in combined_systems.items())), combined_systems.values()):

            _, (velocities_out,) = response()
            self.set_velocities(velocities_out, system_data["indices"])

    def evaluate_energies(self, indices=None):
        """
        Evaluates the potential and kinetic energies of the specified instances.

        Parameters
        ----------
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Returns
        -------
        tuple(array of float)
            For all specified instances, the potential energy of each, and for
            all specified instances, the kinetic energy of each.
        """

        # Store non-deduplicated indices (do not call unique() yet) so that if
        # duplicate indices are provided, duplicate results can be reported.
        indices = self.__resolve_with_size(self.__instance_count, indices)

        # Save the current stack identifiers since we will require each instance
        # to have a dedicated OpenMM system for the energy evaluation, and will
        # therefore need to modify them.
        old_stacks = self.get_stacks()

        try:
            self.set_stacks_separate()

            # Now call unique() on the indices to ensure that each OpenMM system
            # created contains exactly one instance even if duplicate indices
            # are provided.
            combined_systems = self.__get_combined_systems(numpy.unique(indices))

            potential_energies = {}
            kinetic_energies = {}

            for response, system_data in zip(self.__manager._distribute(*(
                support.Arguments(
                    system_data["system_path"],
                    system_data["context_data"],
                    self.__rng.spawn(1)[0],
                    system_data["vectors"],
                    self.get_positions(system_data["indices"]),
                    self.get_velocities(system_data["indices"]),
                    False,
                    False,
                    False,
                    True,
                    False,
                    system_data["enforce_periodic"],
                    system_data["center_coordinates"],
                )
                for stack, system_data in combined_systems.items())), combined_systems.values()):

                # This assumes that set_stacks_separate() sets stacks 0, 1, 2,
                # etc., and that stacks containing single instances will have
                # temperature scales of 1.  This should be the case.
                _, (potential_energy_out, kinetic_energy_out) = response()
                stack_index, = system_data["indices"]
                potential_energies[stack_index] = potential_energy_out
                kinetic_energies[stack_index] = kinetic_energy_out

        finally:
            # Ensure that an exception being raised does not leave the modified
            # stack identifiers in place.
            self.set_stacks(old_stacks)

        return (
            numpy.array([potential_energies[index] for index in indices]),
            numpy.array([kinetic_energies[index] for index in indices]),
        )

    def integrate(self, step_count, write_start=None, write_stop=None, write_step=None, write_velocities=False, write_energies=False, broadcast_energies=False, indices=None):
        """
        Runs molecular dynamics for the specified instances, and updates
        periodic box vector components, position coordinates, and velocity
        components.

        Parameters
        ----------
        step_count : int
            Number of steps over which to integrate.
        write_start : int, optional
            The index of the first step at which to write positions.
        write_stop : int, optional
            The index of the first step at which to cease writing positions.
        write_step : int, optional
            The interval in steps at which to write positions after the first
            step at which positions have been written.
        write_velocities : bool, optional
            Whether or not to write velocities with positions.
        write_energies : bool, optional
            Whether or not to write potential and kinetic energies with positions.
        broadcast_energies : bool, optional
            Whether or not to return potential energies evaluated by
            broadcasting the positions of each instance in turn to all
            instances.  This is only particularly meaningful for replica
            exchange when all instances have identical templates.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Notes
        -----
        If neither ``write_start``, ``write_stop``, nor ``write_step`` are
        given, no writing will occur.  Otherwise, writing will occur at the
        steps corresponding to a slice constructed from the given parameters.

        Returns
        -------
        IntegrationResult or (IntegrationResult, array of float)
            An object containing information about positions written, and, if
            ``broadcast_energies`` is ``True``, an array of potential energies.
        """

        if not isinstance(step_count, int):
            raise TypeError("step_count must be an int")
        step_count = int(step_count)
        if step_count < 0:
            raise ValueError("step_count must be non-negative")

        if write_start is not None:
            if not isinstance(write_start, int):
                raise TypeError("write_start must be an int")
            write_start = int(write_start)
            if write_start < 0:
                raise ValueError("write_start must be non-negative")

        if write_stop is not None:
            if not isinstance(write_stop, int):
                raise TypeError("write_stop must be an int")
            write_stop = int(write_stop)
            if write_stop < 0:
                raise ValueError("write_stop must be non-negative")

        if write_step is not None:
            if not isinstance(write_step, int):
                raise TypeError("write_step must be an int")
            write_step = int(write_step)
            if write_step < 1:
                raise ValueError("write_step must be positive")

        if not isinstance(write_velocities, bool):
            raise TypeError("write_velocities must be a bool")
        write_velocities = bool(write_velocities)

        if not isinstance(write_energies, bool):
            raise TypeError("write_energies must be a bool")
        write_energies = bool(write_energies)

        if not isinstance(broadcast_energies, bool):
            raise TypeError("broadcast_energies must be a bool")
        broadcast_energies = bool(broadcast_energies)

        if write_start is None and write_stop is None and write_step is None:
            write_start = write_stop = 0
            write_step = 1

        # Store non-deduplicated indices so that if duplicate indices are
        # provided and broadcast_energies is set, duplicate results can be
        # reported accurately.
        indices = self.__resolve_with_size(self.__instance_count, indices)
        indices_unique = numpy.unique(indices)
        combined_systems = self.__get_combined_systems(indices_unique)

        integration_results = []

        if broadcast_energies:
            potential_energies = {}

        for response, system_data in zip(self.__manager._distribute(*(
            support.Arguments(
                system_data["system_path"],
                system_data["context_data"],
                self.__rng.spawn(1)[0],
                system_data["vectors"],
                self.get_positions(system_data["indices"]),
                self.get_velocities(system_data["indices"]),
                True,
                True,
                True,
                False,
                broadcast_energies,
                system_data["enforce_periodic"],
                system_data["center_coordinates"],
                (simulation.Command.INTEGRATE, support.Arguments(step_count, write_start, write_stop, write_step, write_velocities, write_energies)),
            )
            for stack, system_data in combined_systems.items())), combined_systems.values()):

            (integration_result,), (vectors_out, positions_out, velocities_out, *broadcast_out) = response()
            self.set_vectors(vectors_out, system_data["indices"])
            self.set_positions(positions_out, system_data["indices"])
            self.set_velocities(velocities_out, system_data["indices"])
            integration_results.append(integration_result)

            if broadcast_energies:
                broadcast_energy_scale = 1 / numpy.sum(1 / system_data["temperature_scales"])
                for instance_index, broadcast_energy in zip(system_data["indices"], *broadcast_out, strict=True):
                    potential_energies[instance_index] = broadcast_energy * broadcast_energy_scale

        expected_frame_count = len(range(step_count + 1)[write_start:write_stop:write_step])
        path_table = {path: path_index for path_index, path in enumerate(sorted(set(path for path, _, _, _ in integration_results)))}

        path_indices = {}
        byte_offsets = {}
        particle_indices_start = {}
        particle_indices_end = {}

        for system_data, (path, byte_offset, frame_count, particle_offsets) in zip(combined_systems.values(), integration_results):
            if frame_count != expected_frame_count:
                raise RuntimeError("unexpected number of frames")

            stack_indices = system_data["indices"]
            if not stack_indices.size + 1 == particle_offsets.size:
                raise RuntimeError("unexpected number of offsets")

            for stack_index, particle_index_start, particle_index_end in zip(stack_indices, particle_offsets[:-1], particle_offsets[1:]):
                path_indices[stack_index] = path_table[path]
                byte_offsets[stack_index] = byte_offset
                particle_indices_start[stack_index] = particle_index_start
                particle_indices_end[stack_index] = particle_index_end

        integration_result = IntegrationResult(
            tuple(path_table), 
            indices_unique,
            expected_frame_count,
            numpy.array([path_indices[index] for index in indices_unique], dtype=int),
            numpy.array([byte_offsets[index] for index in indices_unique], dtype=int),
            numpy.array([particle_indices_start[index] for index in indices_unique], dtype=int),
            numpy.array([particle_indices_end[index] for index in indices_unique], dtype=int),
        )
        if broadcast_energies:
            return integration_result, numpy.array([potential_energies[index] for index in indices])
        else:
            return integration_result
    
    def replica_exchange(self, pair_generator, acceptance_criterion, step_count, result_path, write_start=None, write_stop=None, write_step=None, exchange_start=None, exchange_stop=None, exchange_step=None, write_velocities=False, indices=None):
        """
        Runs replica exchange molecular dynamics for the specified instances,
        and updates periodic box vector components, position coordinates, and
        velocity components.

        Parameters
        ----------
        pair_generator : ExchangePairGenerator
            The algorithm to use for proposing pairs of replicas to exchange.
        acceptance_criterion : AcceptanceCriterion
            The algorithm to use for determining whether or not a proposed swap
            between a pair of replicas should take place.
        step_count : int
            Number of steps over which to integrate.
        result_path : str
            A path to which to write integration results.
        write_start : int, optional
            The index of the first step at which to write positions.
        write_stop : int, optional
            The index of the first step at which to cease writing positions.
        write_step : int, optional
            The interval in steps at which to write positions after the first
            step at which positions have been written.
        exchange_start : int, optional
            The index of the first step at which to perform replica exchange
            attempts.
        exchange_stop : int, optional
            The index of the first step at which to cease performing replica
            exchange attempts.
        exchange_step : int, optional
            The interval in steps at which to perform replica exchange attempts
            after the first step at which replica exchange attempts have been
            performed.
        write_velocities : bool, optional
            Whether or not to write velocities with positions.
        indices : int, slice, array of int, or array of bool, optional
            Specification of instances.  By default, all instances will be
            selected.

        Notes
        -----
        If neither ``write_start``, ``write_stop``, nor ``write_step`` are
        given, no writing will occur.  Otherwise, writing will occur at the
        steps corresponding to a slice constructed from the given parameters.
        Likewise, if neither ``exchange_start``, ``exchange_stop``, nor
        ``exchange_step`` are given, no replica exchange attempts will be
        performed.  Otherwise, replica exchange attempts will be performed at
        the steps corresponding to a slice constructed from these parameters.
        """

        if not isinstance(pair_generator, ExchangePairGenerator):
            raise TypeError("pair_generator must be an ExchangePairGenerator")

        if not isinstance(acceptance_criterion, AcceptanceCriterion):
            raise TypeError("acceptance_criterion must be an AcceptanceCriterion")

        if not isinstance(step_count, int):
            raise TypeError("step_count must be an int")
        step_count = int(step_count)
        if step_count < 0:
            raise ValueError("step_count must be non-negative")

        if not isinstance(result_path, str):
            raise TypeError("result_path must be a str")

        if write_start is not None:
            if not isinstance(write_start, int):
                raise TypeError("write_start must be an int")
            write_start = int(write_start)
            if write_start < 0:
                raise ValueError("write_start must be non-negative")

        if write_stop is not None:
            if not isinstance(write_stop, int):
                raise TypeError("write_stop must be an int")
            write_stop = int(write_stop)
            if write_stop < 0:
                raise ValueError("write_stop must be non-negative")

        if write_step is not None:
            if not isinstance(write_step, int):
                raise TypeError("write_step must be an int")
            write_step = int(write_step)
            if write_step < 1:
                raise ValueError("write_step must be positive")

        if exchange_start is not None:
            if not isinstance(exchange_start, int):
                raise TypeError("exchange_start must be an int")
            exchange_start = int(exchange_start)
            if exchange_start < 0:
                raise ValueError("exchange_start must be non-negative")

        if exchange_stop is not None:
            if not isinstance(exchange_stop, int):
                raise TypeError("exchange_stop must be an int")
            exchange_stop = int(exchange_stop)
            if exchange_stop < 0:
                raise ValueError("exchange_stop must be non-negative")

        if exchange_step is not None:
            if not isinstance(exchange_step, int):
                raise TypeError("exchange_step must be an int")
            exchange_step = int(exchange_step)
            if exchange_step < 1:
                raise ValueError("exchange_step must be positive")

        if not isinstance(write_velocities, bool):
            raise TypeError("write_velocities must be a bool")
        write_velocities = bool(write_velocities)

        if write_start is None and write_stop is None and write_step is None:
            write_start = write_stop = 0
            write_step = 1

        if exchange_start is None and exchange_stop is None and exchange_step is None:
            exchange_start = exchange_stop = 0
            exchange_step = 1

        # Select the unique indices requested.
        indices = numpy.unique(self.__resolve_with_size(self.__instance_count, indices))

        # Prepare to calculate when writing needs to take place.
        write_range = range(step_count + 1)[write_start:write_stop:write_step]

        def slice_write_range(start, stop):
            # Return parameters defining a range that is a subset of write_range
            # having values greater than or equal to start and less than stop,
            # offset by start.
            index_start = max(0, (start - write_range.start + write_range.step - 1) // write_range.step)
            index_stop = min(len(write_range), (stop - write_range.start + write_range.step - 1) // write_range.step)
            sliced_range = write_range[index_start:index_stop]
            return range(sliced_range.start - start, sliced_range.stop - start, sliced_range.step)

        # Keep track of how many steps we have integrated so far.
        step_index = 0

        # TODO: See if indices can be evaluated directly or if broadcasting will be
        # needed.
        raise NotImplementedError

        for exchange_index in range(step_count + 1)[exchange_start:exchange_stop:exchange_step]:
            # See what step we are at, what step we need to get to to make the
            # next exchange attempt, and how many steps we need to simulate to
            # get there.
            integrate_count = exchange_index - step_index
            if integrate_count:
                self.integrate(integrate_count, *slice_write_range(step_index, step_index + integrate_count), write_velocities, ..., ..., indices)
            step_index += integrate_count

            # TODO: Call to perform swapping.
            raise NotImplementedError

        # Finish if we have additional steps to simulate after the last exchange
        # attempt.  Even if we advance zero steps, there may be a final frame to
        # write (hence, the stopping index for slicing the writing range is
        # incremented by one to include it).
        integrate_count = step_count - step_index
        self.integrate(integrate_count, *slice_write_range(step_index, step_index + integrate_count + 1), write_velocities, ..., ..., indices)

        # TODO: Collect and return results.
        raise NotImplementedError

    def __update_instance_data(self):
        # Updates tables used to slice arrays containing per-particle
        # properties.  In particular, when self.__template_indices is changed,
        # this method can check for valid template indices, then update
        # self.__instance_sizes (containing the number of particles in each
        # instance) and self.__instance_offsets (whose consecutive pairs of
        # indices can be used to create ranges or slices including the indices
        # of particles belonging to individual instances).

        if numpy.any((self.__template_indices < 0) | (self.__template_indices >= self.__template_count)):
            # If no templates are present, no valid template indices will exist,
            # so generate an appropriate error message in this case.
            raise ValueError(f"template indices must be non-negative and less than {self.__template_count}"
                if self.__template_count else "no templates exist")

        # Save and return the old offsets before resetting them so that
        # per-particle property arrays can be adjusted appropriately.
        self.__instance_sizes = self.__template_sizes[self.__template_indices]
        old_offsets, self.__instance_offsets = self.__instance_offsets, numpy.concatenate(((0,), numpy.cumsum(self.__instance_sizes)))
        return old_offsets

    def __adjust_per_instance(self, array, padding_value=0):
        # Adjusts the size of the first dimension of a NumPy array to match the
        # current number of instances, by either truncation or padding with
        # a given value.  The array, a slice of it, or a new array, might be
        # returned.

        length = array.shape[0]

        # If no adjustment is necessary, do nothing.
        if self.__instance_count == length:
            return array

        # If the array must shrink, truncate it.
        if self.__instance_count < length:
            return array[:self.__instance_count]

        # Otherwise, the array must grow; create a new array of the given
        # padding value and copy in the items.
        new_array = numpy.full((self.__instance_count, *array.shape[1:]), padding_value, dtype=array.dtype)
        new_array[:length] = array
        return new_array

    def __adjust_per_particle(self, array, old_offsets):
        # Adjusts the size of the first dimension of a NumPy array to match the
        # total number of particles across all instances.  Truncation or padding
        # with zeros will occur as needed on a per-instance basis.  The array,
        # or a new array, might be returned.

        # If no adjustment is necessary, do nothing.
        new_offsets = self.__instance_offsets
        if old_offsets.shape == new_offsets.shape and numpy.all(old_offsets == new_offsets):
            return array

        # Otherwise, create a new array of zeros and copy in the items.
        new_array = numpy.zeros((new_offsets[-1], *array.shape[1:]), dtype=array.dtype)
        for instance_index in range(min(old_offsets.size, new_offsets.size) - 1):
            old_start, old_end = old_offsets[instance_index:instance_index + 2]
            new_start, new_end = new_offsets[instance_index:instance_index + 2]
            min_len = min(old_end - old_start, new_end - new_start)
            new_array[new_start:new_start + min_len] = array[old_start:old_start + min_len]
        return new_array

    def __get_per_instance(self, array, indices):
        return array[self.__resolve_with_size(self.__instance_count, indices)]

    def __get_per_particle(self, array, indices):
        return array[self.__resolve_with_offsets(self.__instance_offsets, indices)]

    def __set_per_instance(self, array, indices, values):
        indices = self.__resolve_with_size(self.__instance_count, indices)
        array[indices] = numpy.broadcast_to(values, (indices.size, *array.shape[1:]))

    def __set_per_particle(self, array, indices, values):
        indices = self.__resolve_with_offsets(self.__instance_offsets, indices)
        array[indices] = numpy.broadcast_to(values, (indices.size, *array.shape[1:]))

    def __get_combined_systems(self, indices):
        # Creates combined OpenMM systems for the instances corresponding to
        # each of the unique indices in the given array based on the current
        # parallel simulation state.

        # Retrieve the stack identifiers indicating the stacks that actually
        # need to have combined systems created from them.
        stack_set = numpy.unique(self.__stacks[indices])

        # Check the selected ensemble type.
        if isinstance(self.__ensemble, CanonicalEnsemble):
            npt = False
        elif isinstance(self.__ensemble, IsothermalIsobaricEnsemble):
            npt = True
        else:
            raise RuntimeError("unrecognized ensemble type")

        # Retrieve whether or not to enforce periodic boundary conditions and
        # center coordinates (these property values can differ per instance).
        enforce_periodic = self.__property_values["enforce_periodic"][indices]
        center_coordinates = self.__property_values["center_coordinates"][indices]

        # Retrieve the step length.  This is done across all stacks so that
        # simulating stacks separately does not advance simulation times non-
        # uniformly.  If no systems are selected, step_length will not be set,
        # but it will not be used as stack_set will be empty and the body of the
        # loop processing each stack will never execute.
        step_length_array = self.__property_values["step_length"][indices]
        if step_length_array.size:
            step_length = step_length_array[0]
            if numpy.any(step_length_array[1:] != step_length):
                warnings.warn("Mismatched step lengths; choosing length from first instance selected", support.MultiOpenMMWarning)

        # Ensure that stacking does not occur for simulations in the
        # isothermal-isobaric ensemble (where instance periodic box vectors may
        # change independently of each other).
        if npt:
            if indices.size != stack_set.size:
                raise support.MultiOpenMMError("simulations in the isothermal-isobaric ensemble may not use stacking")

        # Process each stack identifier.
        combined_systems = {}
        for stack in stack_set:
            mask = self.__stacks[indices] == stack

            # Retrieve vectors for the stack.
            vectors_array = self.__vectors[indices][mask]
            vectors = vectors_array[0]
            if numpy.any(vectors_array[1:] != vectors):
                warnings.warn("Mismatched periodic box vectors; choosing vectors from first instance selected", support.MultiOpenMMWarning)

            # Retrieve the integrator constraint tolerance.  For each stack,
            # there will be at least one instance, so for this and other
            # properties, there will be at least one value.
            constraint_tolerance_array = self.__property_values["constraint_tolerance"][indices][mask]
            constraint_tolerance = constraint_tolerance_array[0]
            if numpy.any(constraint_tolerance_array[1:] != constraint_tolerance):
                warnings.warn("Mismatched constraint tolerances; choosing tolerance from first instance selected", support.MultiOpenMMWarning)

            # Calculate an average run temperature for the combined system so
            # that the scale factors associated with individual instances will
            # be close to 1.
            temperatures = self.__property_values["temperature"][indices][mask]
            if temperatures.size == 1:
                # If the combined system contains exactly one instance, ensure
                # that the run temperature will be exactly equal to the instance
                # temperature (the scipy.stats.gmean implementation does not
                # guarantee this due to internal roundoff: exp(log(x)) != x).
                run_temperature, = temperatures
            else:
                run_temperature = scipy.stats.gmean(temperatures)

            # Retrieve the thermostat characteristic time and calculate the
            # friction coefficient or collision frequency for the thermostat.
            thermostat_steps_array = self.__property_values["thermostat_steps"][indices][mask]
            thermostat_steps = thermostat_steps_array[0]
            if numpy.any(thermostat_steps_array[1:] != thermostat_steps):
                warnings.warn("Mismatched thermostat characteristic times; choosing time from first instance selected", support.MultiOpenMMWarning)
            friction = 1 / (step_length * thermostat_steps)

            set_tolerance_method = ("setConstraintTolerance", support.Arguments(constraint_tolerance))
            runtime_data = []
            forces = []

            match self.__ensemble.thermostat:
                case Thermostat.ANDERSEN:
                    # Indicate that a default temperature is to be used when
                    # creating the thermostat and that the temperature is to be
                    # updated as a context variable.

                    runtime_data.append((
                        simulation.RuntimeUpdateObject.CONTEXT,
                        "setParameter",
                        support.Arguments(openmm.AndersenThermostat.Temperature(), run_temperature),
                    ))
                    integrator_data = simulation.ObjectData(
                        openmm.VerletIntegrator,
                        support.Arguments(step_length),
                        False,
                        set_tolerance_method,
                    )
                    forces.append(simulation.ObjectData(
                        openmm.AndersenThermostat,
                        support.Arguments(1.0, friction),
                        True,
                    ))

                case Thermostat.LANGEVIN:
                    # Indicate that a default temperature is to be used when
                    # creating the integrator and that the temperature is to be
                    # updated directly on the integrator.

                    runtime_data.append((
                        simulation.RuntimeUpdateObject.INTEGRATOR,
                        "setTemperature",
                        support.Arguments(run_temperature),
                    ))
                    integrator_data = simulation.ObjectData(
                        openmm.LangevinMiddleIntegrator,
                        support.Arguments(1.0, friction, step_length),
                        True,
                        set_tolerance_method,
                    )

                case Thermostat.BROWNIAN:
                    # Indicate that a default temperature is to be used when
                    # creating the integrator and that the temperature is to be
                    # updated directly on the integrator.

                    runtime_data.append((
                        simulation.RuntimeUpdateObject.INTEGRATOR,
                        "setTemperature",
                        support.Arguments(run_temperature),
                    ))
                    integrator_data = simulation.ObjectData(
                        openmm.BrownianIntegrator,
                        support.Arguments(1.0, friction, step_length),
                        True,
                        set_tolerance_method,
                    )

                case _:
                    raise RuntimeError("unrecognized Thermostat")

            if npt:
                # In the isothermal-isobaric ensemble, it has already been
                # checked that stacking will not occur, so there will be single
                # values for pressure and the barostat characteristic time.
                pressure, = self.__property_values["pressure"][indices][mask]
                barostat_steps, = self.__property_values["barostat_steps"][indices][mask]

                match self.__ensemble.barostat:
                    case Barostat.MC_ISOTROPIC:
                        # Indicate that a default pressure and temperature are
                        # to be used when creating the barostat and that the
                        # temperature and pressure are to be updated as context
                        # variables.

                        runtime_data.append((
                            simulation.RuntimeUpdateObject.CONTEXT,
                            "setParameter",
                            support.Arguments(openmm.MonteCarloBarostat.Pressure(), pressure),
                        ))
                        runtime_data.append((
                            simulation.RuntimeUpdateObject.CONTEXT,
                            "setParameter",
                            support.Arguments(openmm.MonteCarloBarostat.Temperature(), temperature),
                        ))
                        forces.append(simulation.ObjectData(
                            openmm.MonteCarloBarostat,
                            support.Arguments(1.0, 1.0, barostat_steps),
                            True,
                        ))

                    case _:
                        raise RuntimeError("unrecognized Barostat")
           
            # For each combined system, return a dictionary containing the
            # system and information about it.
            temperature_scales = temperatures / run_temperature
            combined_systems[stack] = dict(
                center_coordinates=center_coordinates[mask],
                context_data=simulation.ContextData(runtime_data, integrator_data, *forces),
                enforce_periodic=enforce_periodic[mask],
                indices=indices[mask],
                run_temperature=run_temperature,
                system_path=self.__get_combined_system(self.__template_indices[indices][mask], temperature_scales),
                temperature_scales=temperature_scales,
                vectors=vectors,
            )

        return combined_systems
    
    def __get_combined_system(self, template_indices, temperature_scales):
        # Creates a combined OpenMM system, pickles it, and stores in the cache
        # index the path to the pickle file for the combined system parameters.

        key = (tuple(template_indices), tuple(temperature_scales))

        if key not in self.__cache_index:
            with tempfile.NamedTemporaryFile(prefix="multiopenmm_system_", suffix=".pickle.gz", dir=self.__cache_directory.name, delete=False) as file:
                with gzip.GzipFile("", "wb", fileobj=file, compresslevel=1, mtime=0) as compressor:
                    pickle.dump(stacking.stack(self.__templates, template_indices, temperature_scales), compressor)
            self.__cache_index[key] = file.name

        return self.__cache_index[key]
    
    @classmethod
    def __resolve_with_size(cls, size, indices):
        # Converts an index, list of indices, array of indices, slice, or mask
        # into a one-dimensional list of indices suitable for indexing an array
        # of the given size.

        # If the specification is None, an array of shape (1, size) will be
        # created and flattened to one of shape (size,), thus including all
        # items in order in the specification as if it had been slice(None).
        return numpy.atleast_1d(numpy.arange(size)[indices]).flatten()

    @classmethod
    def __resolve_with_offsets(cls, offsets, indices):
        # Converts an index, list of indices, array of indices, slice, or mask
        # into a one-dimensional list of indices suitable for indexing blocks of
        # an array delimited by the indices in the given array of offsets.

        # If an empty array is returned by the first resolution, return an empty
        # array to ensure that only non-empty lists of arrays are concatenated.
        indices = cls.__resolve_with_size(offsets.size - 1, indices)
        return numpy.concatenate([numpy.arange(offsets[index], offsets[index + 1]) for index in indices]) if indices.size else indices
    
class Ensemble(abc.ABC):
    """
    Represents a generic ensemble.  Subclasses represent specific ensembles.
    Adds the properties ``enforce_periodic`` (:py:class:`bool`),
    ``center_coordinates`` (:py:class:`bool`), ``step_length``
    (:py:class:`float`), and ``constraint_tolerance`` (:py:class:`float`) to
    molecular system instances.  Here, ``enforce_periodic`` controls whether or
    not periodic boundary conditions are applied to coordinates retrieved from
    OpenMM contexts (this will usually be desired when one or more forces use
    periodic boundary conditions), and ``center_coordinates`` controls whether
    or not coordinates retrieved from OpenMM contexts will be adjusted to place
    instance centers of mass at the origin (this will usually be desired when no
    forces use periodic boundary conditions).  ``step_length`` controls the time
    elapsed per integration step, while ``constraint_tolerance`` adjusts the
    tolerance to which position coordinates and velocity components should be
    held to satisfy rigid constraints.
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def property_types(self):
        return {
            "enforce_periodic": bool,
            "center_coordinates": bool,
            "step_length": float,
            "constraint_tolerance": float,
        }

    @property
    @abc.abstractmethod
    def property_defaults(self):
        return {
            "enforce_periodic": False,
            "center_coordinates": False,
            "step_length": DEFAULT_STEP_LENGTH,
            "constraint_tolerance": DEFAULT_CONSTRAINT_TOLERANCE,
        }

class CanonicalEnsemble(Ensemble):
    """
    Represents a canonical ensemble maintained by a thermostat.  Adds the
    properties ``temperature`` (:py:class:`float`) and ``thermostat_steps``
    (:py:class:`float`) to molecular system instances.  Here,
    ``thermostat_steps`` determines the characteristic time of the thermostat
    (*i.e.*, the reciprocal of the collision frequency for an Andersen
    thermostat, or the reciprocal of the friction coefficient for Langevin or
    Brownian dynamics) in units of integration steps.

    Parameters
    ----------
    thermostat : multiopenmm.Thermostat, optional
        The thermostat algorithm to use.  By default, an Andersen thermostat
        will be used.
    """

    __slots__ = ("__thermostat",)

    def __init__(self, thermostat=Thermostat.ANDERSEN):
        super().__init__()

        if not isinstance(thermostat, Thermostat):
            raise TypeError("thermostat must be a Thermostat")

        self.__thermostat = Thermostat(thermostat)

    @property
    def thermostat(self):
        """
        multiopenmm.Thermostat: The thermostat algorithm to use.
        """

        return self.__thermostat

    @property
    def property_types(self):
        return {
            **super().property_types,
            "temperature": float,
            "thermostat_steps": float,
        }

    @property
    def property_defaults(self):
        return {
            **super().property_defaults,
            "temperature": DEFAULT_TEMPERATURE,
            "thermostat_steps": DEFAULT_THERMOSTAT_STEPS,
        }

class IsothermalIsobaricEnsemble(Ensemble):
    """
    Represents an isothermal-isobaric ensemble maintained by a thermostat and a
    barostat.  Adds the properties ``temperature`` (:py:class:`float`),
    ``pressure`` (:py:class:`float`), ``thermostat_steps`` (:py:class:`float`),
    and ``barostat_steps`` (:py:class:`int`) to molecular system instances.
    Here, ``thermostat_steps`` determines the characteristic time of the
    thermostat (*i.e.*, the reciprocal of the collision frequency for an
    Andersen thermostat, or the reciprocal of the friction coefficient for
    Langevin or Brownian dynamics) in units of integration steps, and
    ``barostat_steps`` likewise controls the frequency at which the barostat can
    adjust the volumes of instances.

    Parameters
    ----------
    thermostat : multiopenmm.Thermostat, optional
        The thermostat algorithm to use.  By default, an Andersen thermostat
        will be used.
    barostat : multiopenmm.Barostat, optional
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

        self.__thermostat = Thermostat(thermostat)
        self.__barostat = Barostat(barostat)

    @property
    def thermostat(self):
        """
        multiopenmm.Thermostat: The thermostat algorithm to use.
        """

        return self.__thermostat

    @property
    def barostat(self):
        """
        multiopenmm.Barostat: The barostat algorithm to use.
        """

        return self.__barostat

    @property
    def property_types(self):
        return {
            **super().property_types,
            "temperature": float,
            "pressure": float,
            "thermostat_steps": float,
            "barostat_steps": int,
        }

    @property
    def property_defaults(self):
        return {
            **super().property_defaults,
            "temperature": DEFAULT_TEMPERATURE,
            "pressure": DEFAULT_PRESSURE,
            "thermostat_steps": DEFAULT_THERMOSTAT_STEPS,
            "barostat_steps": DEFAULT_BAROSTAT_STEPS,
        }

class IntegrationResult:
    """
    Holds details of trajectory frames written during calls to
    :py:meth:`multiopenmm.Simulation.integrate`.
    """

    __slots__ = ("__path_table", "__indices", "__frame_count", "__path_indices",
        "__byte_offsets", "__particle_indices_start", "__particle_indices_end")

    def __init__(self, *data):
        self._set_data(*data)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self._get_data() == other._get_data()

    def __getstate__(self):
        return self._get_data()

    def __setstate__(self, state):
        self._set_data(*state)

    def _get_data(self):
        return (
            self.__path_table,
            self.__indices,
            self.__frame_count,
            self.__path_indices,
            self.__byte_offsets,
            self.__particle_indices_start,
            self.__particle_indices_end,
        )

    def _set_data(self, *data):
        (
            self.__path_table,
            self.__indices,
            self.__frame_count,
            self.__path_indices,
            self.__byte_offsets,
            self.__particle_indices_start,
            self.__particle_indices_end,
        ) = data

class ExchangePairGenerator(abc.ABC):
    """
    Defines an algorithm for generating pairs of instance indices for replica
    exchange swap attempts.
    """

    __slots__ = ()

    @abc.abstractmethod
    def generate(self, rng, iteration_index, instance_count):
        """
        Generates pairs of indices of instances for which replica exchange swaps
        should be attempted.

        Parameters
        ----------
        rng : numpy.random.Generator
            A source of randomness that should be used if the implemented
            algorithm for generating pairs requires randomness.
        iteration_index : int
            An index that will be incremented for every new set of swap
            attempts.
        instance_count : int
            The total number of instances undergoing replica exchange.

        Returns
        -------
        iterable of (int, int)
            Indices specifying pairs of instances for which replica exchange
            swaps should be attempted.

        Notes
        -----
        This method must be implemented in derived classes.  It should be
        assumed that instance indices increase sequentially from ``0`` to
        ``iteration_index - 1``.
        """

        raise NotImplementedError

class RandomAdjacentExchangePairGenerator(ExchangePairGenerator):
    """
    Randomly generates pairs of adjacent instance indices for replica exchange
    swap attempts.

    Parameters
    ----------
    swap_count : int
        The number of swaps to attempt (*i.e.*, the number of pairs to generate
        per call to :py:meth:`generate`).
    with_replacement : bool, optional
        Whether or not the same swap should be attempted more than once in a set
        (``False`` by default).  If not, ``swap_count`` must be less than the
        number of instances undergoing replica exchange.
    """

    __slots__ = ("__swap_count", "__with_replacement")

    def __init__(self, swap_count, with_replacement=False):
        if not isinstance(swap_count, int):
            raise TypeError("swap_count must be an int")
        swap_count = int(swap_count)
        if swap_count < 0:
            raise ValueError("swap_count must be non-negative")

        if not isinstance(with_replacement, bool):
            raise TypeError("with_replacement must be a bool")
        with_replacement = bool(with_replacement)

        self.__swap_count = swap_count
        self.__with_replacement = with_replacement

    @property
    def swap_count(self):
        """
        int : The number of swaps to attempt.
        """

        return self.__swap_count

    @property
    def with_replacement(self):
        """
        bool : Whether or not the same swap should be attempted more than once
        in a set.
        """

        return self.__with_replacement

    def generate(self, rng, iteration_index, instance_count):
        """
        Generates pairs of indices of instances for which replica exchange swaps
        should be attempted.
        """

        for instance_index in rng.choice(instance_count - 1, size=self.__swap_count, replace=self.__with_replacement):
            yield (instance_index, instance_index + 1)

class AcceptanceCriterion(abc.ABC):
    """
    Defines an algorithm for deciding whether or not to accept a replica
    exchange swap.
    """

    __slots__ = ()

    @abc.abstractmethod
    def probability(self, use_pressure, beta_i, beta_j, u_i, u_j, p_i, p_j, v_i, v_j):
        """
        Calculates the probability that a replica exchange swap between two
        replicas (:math:`i` and :math:`j`) should occur.

        Parameters
        ----------
        use_pressure : bool
            ``True`` if a pressure/volume term should be included in the
            acceptance criterion using the given values, ``False`` if the
            pressure and volume values should be ignored and such a term should
            not be included.
        beta_i : float
            :math:`\\beta_i=1/k_BT_i` where :math:`T_i` is the temperature of
            replica :math:`i`.
        beta_j : float
            :math:`\\beta_j=1/k_BT_j` where :math:`T_j` is the temperature of
            replica :math:`j`.
        u_i : float
            :math:`U_i`, the potential energy of replica :math:`i`.
        u_j : float
            :math:`U_j`, the potential energy of replica :math:`j`.
        p_i : float
            :math:`P_i`, the pressure in replica :math:`i`.
        p_j : float
            :math:`P_j`, the pressure in replica :math:`j`.
        v_i : float
            :math:`V_i`, the volume of replica :math:`i`.
        v_j : float
            :math:`V_j`, the volume of replica :math:`j`.

        Returns
        -------
        float
            The probability that the swap should be carried out.  This value
            should normally be between 0 and 1, although values less than 0 will
            be interpreted as 0 (*i.e.*, the swap will never occur under any
            condition) and values greater than 1 will be interpreted as 1
            (*i.e.*, the swap will occur unconditionally).
        """

        raise NotImplementedError

class MetropolisAcceptanceCriterion(AcceptanceCriterion):
    """
    An acceptance criterion that calculates the probability of whether or not a
    replica exchange swap between two replicas (:math:`i` and :math:`j`) should
    occur as:

    .. math::
       \\mathcal{P}_\\mathrm{acc}(ij\\rightarrow ji)=\\exp\\left(\\min\\left[0,(
       \\beta_2-\\beta_1)(U_2-U_1)+(\\beta_2P_2-\\beta_1P_1)(V_2-V_1)\\right]
       \\right)

    (for isothermal-isobaric ensemble simulations; for canonical ensemble
    simulations, the second term including the pressures will be absent).
    """

    __slots__ = ()

    def probability(self, use_pressure, beta_i, beta_j, u_i, u_j, p_i, p_j, v_i, v_j):
        """
        Calculates the probability that a replica exchange swap between two
        replicas (:math:`i` and :math:`j`) should occur.
        """

        return numpy.exp(min(0, (beta_j - beta_i) * (u_j - u_i) + ((beta_j * p_j - beta_i * p_i) * (v_j - v_i) if use_pressure else 0)))
