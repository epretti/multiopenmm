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

import enum
import gzip
import numpy
import openmm
import pickle
import tempfile

from . import support

class Command(enum.Enum):
    # Specifies an OpenMM context-related command to execute.

    # Applies constraints to positions or velocities, or both.
    APPLY_CONSTRAINTS = enum.auto()

    # Performs energy minimization.
    MINIMIZE = enum.auto()

    # Assigns velocities from a Maxwell-Boltzmann distribution.
    MAXWELL_BOLTZMANN = enum.auto()

    # Performs time integration.
    INTEGRATE = enum.auto()

class RuntimeUpdateObject(enum.Enum):
    # Specifies an object to perform an action on.

    # Performs an action on a context.
    CONTEXT = enum.auto()

    # Performs an action on an integrator.
    INTEGRATOR = enum.auto()

class ObjectData:
    # Contains instructions for creating an OpenMM object.

    __slots__ = ("__initializer", "__initializer_arguments",
        "__set_random_seed", "__methods")

    def __init__(self, initializer, initializer_arguments, set_random_seed, *methods):
        self.__initializer = initializer
        self.__initializer_arguments = initializer_arguments
        self.__set_random_seed = set_random_seed
        self.__methods = methods

    def __repr__(self):
        repr_list = ", ".join(map(repr, (self.__initializer, self.__initializer_arguments, self.__set_random_seed) + self.__methods))
        return f"ObjectData({repr_list})"

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return (self.__initializer, self.__initializer_arguments, self.__set_random_seed, self.__methods) == \
            (other.__initializer, other.__initializer_arguments, other.__set_random_seed, other.__methods)

    def create(self, rng):
        # Creates the OpenMM object using the stored instructions.

        result = self.__initializer_arguments.apply_to(self.__initializer)
        for method_name, method_arguments in self.__methods:
            method_arguments.apply_to(getattr(result, method_name))
        if self.__set_random_seed:
            result.setRandomNumberSeed(support.get_seed(rng))
        return result

class ContextData:
    # Contains instructions for adding forces to an OpenMM system and creating
    # an integrator and a context.

    __slots__ = ("__runtime_data", "__integrator_data", "__forces")

    def __init__(self, runtime_data, integrator_data, *forces):
        self.__runtime_data = runtime_data
        self.__integrator_data = integrator_data
        self.__forces = forces

    def __repr__(self):
        repr_list = ", ".join(map(repr, (self.__runtime_data, self.__integrator_data) + self.__forces))
        return f"ContextData({repr_list})"

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return (self.__integrator_data, self.__forces) == (other.__integrator_data, other.__forces)

    def create(self, system, platform_data, rng):
        # Adds forces to the given system and creates an integrator and a
        # context using the given platform data.

        for force_data in self.__forces:
            system.addForce(force_data.create(rng))
        integrator = self.__integrator_data.create(rng)
        
        if platform_data is None:
            context = openmm.Context(system, integrator)
        else:
            context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName(platform_data.platform_name), platform_data.platform_properties)

        return context

    @property
    def runtime_data(self):
        # Some data that can be squirrelled away in the ContextData object and
        # modified without causing the ContextData object to compare unequal to
        # an otherwise identical ContextData object.  This is useful to be able
        # to update the simulation thermodynamic conditions without triggering a
        # reconstruction of an entire context from scratch.

        return self.__runtime_data

    @runtime_data.setter
    def runtime_data(self, runtime_data):
        self.__runtime_data = runtime_data

class PlatformData:
    """
    Stores OpenMM platform information that can be used to customize OpenMM
    context creation.

    Parameters
    ----------
    platform_name : str
        The name of the platform to use when creating an OpenMM context.  This
        must be the name of one of the platforms that can be returned by
        :py:meth:`openmm.openmm.Platform.getPlatform`.
    platform_properties : dict(str, str), optional
        Name-value pairs specifying platform-specific properties that should be
        set.  Each property name must be one of the names returned by
        :py:meth:`openmm.openmm.Platform.getPropertyNames` for the platform
        specified by ``platform_name``.  If omitted, no platform-specific
        properties will be stored.
    """

    __slots__ = ("__platform_name", "__platform_properties")

    def __init__(self, platform_name, platform_properties=None):
        if not isinstance(platform_name, str):
            raise TypeError("platform_name must be a str")
        platform_name = str(platform_name)

        platform_properties_items = []
        if platform_properties is not None:
            if not isinstance(platform_properties, dict):
                raise TypeError("platform_properties must be a dict")

            for property_name, property_value in platform_properties.items():
                if not isinstance(property_name, str):
                    raise TypeError("platform property name must be a str")
                if not isinstance(property_value, str):
                    raise TypeError("platform property value must be a str")

                platform_properties_items.append((str(property_name), str(property_value)))

        self.__platform_name = platform_name
        self.__platform_properties = dict(platform_properties_items)

    def __repr__(self):
        return f"PlatformData({self.__platform_name!r}, {self.__platform_properties!r})"

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return (self.__platform_name, self.__platform_properties) == (other.__platform_name, other.__platform_properties)

    @property
    def platform_name(self):
        """
        str: The name of the platform.
        """

        return self.__platform_name

    @property
    def platform_properties(self):
        """
        dict(str, str): Platform-specific properties.
        """

        return dict(self.__platform_properties)

class Client:
    # Maintains client state and executes instructions for performing
    # simulations.

    __slots__ = ("__system_path", "__context_data", "__platform_data",
        "__runtime_data", "__context", "__particle_offsets",
        "__particle_masses", "__rng", "__traj_file")

    def __init__(self):
        self.__system_path = None
        self.__context_data = None
        self.__platform_data = None
        self.__runtime_data = None
        self.__context = None
        self.__particle_offsets = None
        self.__particle_masses = None
        self.__rng = None
        self.__traj_file = None

    def execute(self, system_path, context_data, rng,
            vectors_in, positions_in, velocities_in,
            vectors_out, positions_out, velocities_out, energies_out, broadcast_out,
            enforce_periodic, center_coordinates,
            *commands, platform_data):

        # Reload the system and context if necessary.
        if (self.__system_path, self.__context_data, self.__platform_data) != (system_path, context_data, platform_data):
            # Delete the old context and all of its associated objects.
            del self.__context

            # Reload the system from disk and create a new context.
            with gzip.open(system_path) as file:
                system, particle_offsets = pickle.load(file)
            context = context_data.create(system, platform_data, rng)

            # Update system- and context-associated data.  Clear any stored data
            # in the __runtime_data attribute specifying commands that were
            # executed to change the state of the old context; since we have
            # created a new context, we will unconditionally need to execute the
            # latest set of specified commands on it to set up its state.
            self.__system_path = system_path
            self.__context_data = context_data
            self.__platform_data = platform_data
            self.__runtime_data = None
            self.__context = context
            self.__particle_offsets = particle_offsets
            self.__particle_masses = numpy.array([
                support.strip_units(system.getParticleMass(particle_index))
                for particle_index in range(system.getNumParticles())
            ])

        # Save the random number generator in case one is needed.
        self.__rng = rng

        # Update the context if necessary.  Note that the runtime_data attribute
        # of the ContextData class is not considered as part of an equality
        # comparison, so even if the stored and received ContextData instances
        # compared equal, there may still be differences in the runtime_data.
        if self.__runtime_data != self.__context_data.runtime_data:
            for update_object, method_name, method_arguments in self.__context_data.runtime_data:
                match update_object:
                    case RuntimeUpdateObject.CONTEXT:
                        target_object = self.__context
                    case RuntimeUpdateObject.INTEGRATOR:
                        target_object = self.__context.getIntegrator()
                    case _:
                        raise RuntimeError("unrecognized RuntimeUpdateObject")

                method_arguments.apply_to(getattr(target_object, method_name))

            # Store the specification of the commands executed to update the
            # state of the context so that they need not be executed again
            # unless they change.
            self.__runtime_data = self.__context_data.runtime_data

        # Update context state if given.
        if vectors_in is not None:
            self.__context.setPeriodicBoxVectors(*vectors_in)
        if positions_in is not None:
            self.__context.setPositions(positions_in)
        if velocities_in is not None:
            self.__context.setVelocities(velocities_in)

        # Execute commands.
        command_results = []
        for command, arguments in commands:
            match command:
                case Command.APPLY_CONSTRAINTS:
                    command_result = arguments.apply_to(self.apply_constraints)
                case Command.MINIMIZE:
                    command_result = arguments.apply_to(self.minimize)
                case Command.MAXWELL_BOLTZMANN:
                    command_result = arguments.apply_to(self.maxwell_boltzmann)
                case Command.INTEGRATE:
                    command_result = arguments.apply_to(self.integrate)
                case _:
                    raise RuntimeError("unrecognized Command")
            command_results.append(command_result)

        # Retrieve context state if requested.
        state_results = []

        if vectors_out or positions_out or velocities_out or energies_out or broadcast_out:
            # If there is only one instance encompassing all particles, there is
            # no need to perform special broadcasting logic.  Otherwise,
            # particle positions will need to be broadcasted across instances to
            # evaluate the energies of each instance in turn.
            broadcast_flag = broadcast_out and not (self.__particle_offsets.size == 2
                and not self.__particle_offsets[0] and self.__particle_offsets[1] == self.__particle_masses.size)

            # Determine whether or not the first call to getState() should
            # enforce periodic boundary conditions.  Note that if we have the
            # broadcast_flag flag set, we want to not enforce periodic boundary
            # conditions because we will want to evaluate and restore using
            # unwrapped coordinates for broadcasted potential energy evaluation.
            enforce_periodic_flag = bool(positions_out and not broadcast_flag and numpy.all(enforce_periodic))

            state = self.__context.getState(
                getPositions=positions_out or broadcast_flag,
                getVelocities=velocities_out,
                getEnergy=energies_out or (broadcast_out and not broadcast_flag),
                enforcePeriodicBox=enforce_periodic_flag,
            )

        else:
            broadcast_flag = False

        if vectors_out:
            state_results.append(support.strip_units(state.getPeriodicBoxVectors(asNumpy=True)))

        if positions_out or broadcast_flag:
            # The raw_positions array will never be modified and contains
            # unwrapped coordinates unless broadcast_flag is False and all
            # instances are requesting wrapped coordinates.
            raw_positions = support.strip_units(state.getPositions(asNumpy=True))

        if positions_out:
            # The positions array may be modified by replacing the unwrapped
            # coordinates of some instances with wrapped coordinates, so copy it
            # from the raw_positions array.
            positions = numpy.array(raw_positions)

            # Determine whether or not an extra call to getState() needs to be
            # made to enforce periodic boundary conditions (the case if
            # broadcast_flag is set and some or all instances are requesting
            # wrapped coordinates, or otherwise, some instances are requesting
            # wrapped coordinates and others are not: in either case,
            # enforce_periodic_flag will have been set to False, so we know for
            # sure that we will need wrapped coordinates for at least some
            # instances).
            get_wrapped_state_flag = bool(not enforce_periodic_flag and numpy.any(enforce_periodic))

            # See if we need to get wrapped coordinates in addition to unwrapped
            # coordinates already retrieved.
            if get_wrapped_state_flag:
                wrapped_state = self.__context.getState(getPositions=True, enforcePeriodicBox=True)
                wrapped_positions = support.strip_units(wrapped_state.getPositions(asNumpy=True))
                for instance_index, enforce_periodic_instance in enumerate(enforce_periodic):
                    if enforce_periodic_instance:
                        instance_slice = slice(self.__particle_offsets[instance_index], self.__particle_offsets[instance_index + 1])
                        positions[instance_slice] = wrapped_positions[instance_slice]

            # See if we need to translate any instance coordinates by the
            # instance centers of masses.
            for instance_index, center_coordinates_instance in enumerate(center_coordinates):
                if center_coordinates_instance:
                    instance_slice = slice(self.__particle_offsets[instance_index], self.__particle_offsets[instance_index + 1])
                    instance_masses = self.__particle_masses[instance_slice]
                    if not numpy.any(instance_masses):
                        instance_masses = numpy.ones_like(instance_masses)
                    positions[instance_slice] -= numpy.sum(instance_masses[:, None] * positions[instance_slice], axis=0) / numpy.sum(instance_masses)

            state_results.append(positions)

        if velocities_out:
            state_results.append(support.strip_units(state.getVelocities(asNumpy=True)))

        if energies_out:
            state_results.append(support.strip_units(state.getPotentialEnergy()))
            state_results.append(support.strip_units(state.getKineticEnergy()))

        if broadcast_out:
            broadcast_energies = []

            if broadcast_flag:
                # For each instance, copy positions to all other instances and
                # evaluate the system energy.  If instances have different
                # sizes, this will either select only the first positions from
                # the source instance, or leave extra positions at the
                # destination instance.  Normally, this will be done with
                # identical instances, so no such mismatches will occur (as the
                # energies returned in the case of such mismatches will usually
                # correspond to non-physical configurations); note in this case
                # that if the unscaled energy of an instance is desired, it is
                # necessary to divide the returned energies by the sum of the
                # scale factors for the instances.
                for evaluate_instance_index, (evaluate_start_index, evaluate_end_index) in enumerate(zip(self.__particle_offsets[:-1], self.__particle_offsets[1:])):
                    evaluate_positions = numpy.array(raw_positions)
                    for source_instance_index, (source_start_index, source_end_index) in enumerate(zip(self.__particle_offsets[:-1], self.__particle_offsets[1:])):
                        if source_instance_index == evaluate_instance_index:
                            continue
                        length = min(evaluate_end_index - evaluate_start_index, source_end_index - source_start_index)
                        evaluate_positions[evaluate_start_index:evaluate_start_index + length] = evaluate_positions[source_start_index:source_start_index + length]
                    self.__context.setPositions(evaluate_positions)
                    broadcast_energies.append(support.strip_units(self.__context.getState(getEnergy=True).getPotentialEnergy()))

                # Restore the original positions.
                self.__context.setPositions(raw_positions)

            else:
                # If broadcast_out is set but broadcast_flag is not, it is
                # because there is a single instance and the system energy can
                # just be used directly.
                broadcast_energies.append(support.strip_units(state.getPotentialEnergy()))

            state_results.append(numpy.array(broadcast_energies))

        return command_results, state_results

    def apply_constraints(self, positions, velocities):
        integrator = self.__context.getIntegrator()

        if positions:
            self.__context.applyConstraints(integrator.getConstraintTolerance())
        if velocities:
            self.__context.applyVelocityConstraints(integrator.getConstraintTolerance())

    def minimize(self, tolerance, iteration_count):
        openmm.LocalEnergyMinimizer.minimize(self.__context, tolerance, iteration_count)

    def maxwell_boltzmann(self, run_temperature):
        self.__context.setVelocitiesToTemperature(run_temperature, support.get_seed(self.__rng))

    def integrate(self, step_count, write_start, write_stop, write_step, write_velocities, write_energies):
        integrator = self.__context.getIntegrator()

        step_index = 0
        write_pointer = 0
        write_count = 0

        for write_index in range(step_count + 1)[write_start:write_stop:write_step]:
            # See what step we are at, what step we need to get to to make the
            # next write, and how many steps we need to simulate to get there.
            integrate_count = write_index - step_index
            if integrate_count:
                integrator.step(integrate_count)
            step_index += integrate_count

            # If this is the first write, make sure that we have a trajectory
            # file open and that we record the point in the trajectory file at
            # which the writing is starting.
            if not write_count:
                if self.__traj_file is None:
                    self.__traj_file = tempfile.NamedTemporaryFile(prefix="multiopenmm_data_", suffix=".mmmraw", dir=support.get_scratch_directory(), delete=False)
                    support.RawFileIO.write_header(self.__traj_file)
                    self.__traj_file.flush()
                write_pointer = self.__traj_file.tell()

            # Retrieve vectors and positions to write to the trajectory file.
            state = self.__context.getState(getPositions=True, getVelocities=write_velocities, getEnergy=write_energies)
            support.RawFileIO.write_frame(self.__traj_file,
                vectors=support.strip_units(state.getPeriodicBoxVectors(asNumpy=True)),
                positions=support.strip_units(state.getPositions(asNumpy=True)),
                velocities=support.strip_units(state.getVelocities(asNumpy=True)) if write_velocities else None,
                potential_energy=support.strip_units(state.getPotentialEnergy()) if write_energies else None,
                kinetic_energy=support.strip_units(state.getKineticEnergy()) if write_energies else None,
            )

            write_count += 1

        # Make sure that, upon returning, the expected number of frames will
        # have been written completely to the trajectory file.
        if write_count:
            self.__traj_file.flush()

        # Finish if we have additional steps to simulate after the last write.
        integrate_count = step_count - step_index
        if integrate_count:
            integrator.step(integrate_count)

        return None if self.__traj_file is None else self.__traj_file.name, write_pointer, write_count, self.__particle_offsets
