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
import io
import numpy
import openmm
import pickle

from . import concurrent

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
        platform or platforms used to run the simulations are deterministic.
        For more information, consult the
        `OpenMM user guide <http://docs.openmm.org/latest/userguide/library/04_platform_specifics.html#determinism>`_.
    """

    __slots__ = ("__templates", "__manager", "__ensemble", "__precision",
        "__rng", "__template_count", "__template_sizes", "__property_types",
        "__type", "__instance_count", "__template_indices", "__instance_sizes",
        "__instance_offsets", "__stacks", "__vectors", "__positions",
        "__velocities", "__property_values")

    def __init__(self, templates, manager, ensemble, precision=Precision.DOUBLE, seed=None):
        # Make copies of the system objects to ensure that external changes to
        # them do not invalidate any of the data that we extract from them here.
        self.__templates = []
        for template in templates:
            if not isinstance(template, openmm.System):
                raise TypeError("template must be an OpenMM System")
            self.__templates.append(openmm.XmlSerializer.clone(template))

        if not isinstance(manager, concurrent.Manager):
            raise TypeError("manager must be a Manager")
        if not isinstance(ensemble, Ensemble):
            raise TypeError("ensemble must be an Ensemble")
        if not isinstance(precision, Precision):
            raise TypeError("precision must be a Precision")

        self.__manager = manager
        self.__ensemble = ensemble
        self.__precision = precision

        self.__rng = numpy.random.default_rng(seed)

        self.__template_count = len(self.__templates)
        self.__template_sizes = numpy.array([template.getNumParticles() for template in self.__templates], dtype=int)
        self.__property_types = self.__ensemble.property_types
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
    def instance_count(self):
        """
        int: The number of molecular system instances.

        If this value is increased, new instances will be added after the
        existing instances.  If it is decreased, instances at the end of the
        sequence of existing instances will be removed.  Newly added instances
        will have their properties set to zero.
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
        self.__vectors = self.__adjust_per_instance(self.__vectors)
        self.__positions = self.__adjust_per_particle(self.__positions, old_offsets)
        self.__velocities = self.__adjust_per_particle(self.__velocities, old_offsets)
        
        for name in tuple(self.__property_values.keys()):
            self.__property_values[name] = self.__adjust_per_instance(self.__property_values[name])

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
            For each specified instance, its `(3, 3)` matrix of periodic box
            vector components, in column vector form.

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
            For each specified instance, its `(instance_size, 3)` position
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
            For each specified instance, its `(instance_size, 3)` velocity
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
        Updates stack indices for the specified instances.

        Parameters
        ----------
        stacks : int or array of int
            Stack indices for all or each of the specified instances.  This
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
            Periodic box vector components, in column vector form, for all or
            each of the specified instances.  This parameter will be broadcast
            to an appropriate shape for the instance specification.
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
        minimize
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
        maxwell_boltzmann
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

        # TODO: Implementation.
        raise NotImplementedError

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
       
        # TODO: Implementation.
        raise NotImplementedError

    def integrate(self, step_count, swap_spec, write_spec):
        # TODO: Specification of selection criterion, exchange criterion,
        # exchange frequency, save format, save frequency, etc.  Documentation
        # and implementation.
        raise NotImplementedError

    def __update_instance_data(self):
        # Updates tables used to slice arrays containing per-particle properties.
        # In particular, when self.__template_indices is changed, this method can check
        # for valid template indices, then update self.__instance_sizes
        # (containing the number of particles in each instance) and
        # self.__instance_offsets (whose consecutive pairs of indices can be
        # used to create ranges or slices including the indices of particles
        # belonging to individual instances).

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

    def __adjust_per_instance(self, array):
        # Adjusts the size of the first dimension of a NumPy array to match the
        # current number of instances, by either truncation or padding with
        # zeros.  The array, a slice of it, or a new array, might be returned.

        length = array.shape[0]

        # If no adjustment is necessary, do nothing.
        if self.__instance_count == length:
            return array

        # If the array must shrink, truncate it.
        if self.__instance_count < length:
            return array[:self.__instance_count]

        # Otherwise, the array must grow; create a new array of zeros and copy
        # in the items.
        new_array = numpy.zeros((self.__instance_count, *array.shape[1:]), dtype=array.dtype)
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
    Adds the property ``step_length`` (:py:class:`float`) to molecular system
    instances.
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
    (:py:class:`float`) to molecular system instances.

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

        self.__thermostat = thermostat

    @property
    def thermostat(self):
        """
        multiopenmm.Thermostat: The thermostat algorithm to use.
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
    and ``barostat_steps`` (:py:class:`int`) to molecular system instances.

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

        self.__thermostat = thermostat
        self.__barostat = barostat

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
        return {**super().property_types, "temperature": float, "pressure": float, "thermostat_steps": float, "barostat_steps": int}
