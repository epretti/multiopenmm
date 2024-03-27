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

class ObjectData:
    # Contains instructions for creating an OpenMM object.

    __slots__ = ("__initializer", "__initializer_arguments",
        "__set_random_seed", "__methods")

    def __init__(self, initializer, initializer_arguments, set_random_seed, *methods):
        self.__initializer = initializer
        self.__initializer_arguments = initializer_arguments
        self.__set_random_seed = set_random_seed
        self.__methods = methods

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

    __slots__ = ("__integrator_data", "__forces")

    def __init__(self, integrator_data, *forces):
        self.__integrator_data = integrator_data
        self.__forces = forces

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
            context = openmm.Context(system, integrator, platform_data.platform_name, platform_data.platform_properties)

        return integrator, context

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
    platform_properties : dict(str, str)
        Name-value pairs specifying platform-specific properties that should be
        set.  Each property name must be one of the names returned by
        :py:meth:`openmm.openmm.Platform.getPropertyNames` for the platform
        specified by ``platform_name``.
    """

    __slots__ = ("__platform_name", "__platform_properties")

    def __init__(self, platform_name, platform_properties=None):
        if not isinstance(platform_name, str):
            raise TypeError("platform_name must be a str")

        if platform_properties is not None:
            if not isinstance(platform_properties, dict):
                raise TypeError("platform_properties must be a dict")

            for property_name, property_value in platform_properties:
                if not isinstance(property_name, str):
                    raise TypeError("platform property name must be a str")
                if not isinstance(property_value, str):
                    raise TypeError("platform property value must be a str")

            platform_properties = dict(platform_properties)

        self.__platform_name = platform_name
        self.__platform_properties = platform_properties

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

    __slots__ = ("__system_path", "__context_data", "__integrator", "__context",
        "__particle_offsets", "__particle_masses", "__rng", "__traj_file")

    def __init__(self):
        self.__system_path = None
        self.__context_data = None
        self.__integrator = None
        self.__context = None
        self.__particle_offsets = None
        self.__particle_masses = None
        self.__rng = None
        self.__traj_file = None


    def execute(self, system_path, context_data, rng,
            vectors_in, positions_in, velocities_in,
            vectors_out, positions_out, velocities_out, energies_out,
            enforce_periodic, center_coordinates,
            *commands, platform_data):

        # Reload the system and context if necessary.
        if self.__system_path != system_path or self.__context_data != context_data:
            # Delete the old integrator and context.
            del self.__integrator
            del self.__context

            # Reload the system from disk and create a new integrator and
            # context.
            with gzip.open(system_path) as file:
                system, particle_offsets = pickle.load(file)
            integrator, context = context_data.create(system, platform_data, rng)

            # Update system- and context-associated data.
            self.__system_path = system_path
            self.__context_data = context_data
            self.__integrator = integrator
            self.__context = context
            self.__particle_offsets = particle_offsets
            self.__particle_masses = numpy.array([
                system.getParticleMass(particle_index).value_in_unit_system(openmm.unit.md_unit_system)
                for particle_index in range(system.getNumParticles())
            ])

        # Save the random number generator in case one is needed.
        self.__rng = rng

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

        if vectors_out or positions_out or velocities_out or energies_out:
            # Determine whether or not the first call to getState() should
            # enforce periodic boundary conditions.
            enforce_periodic_flag = bool(positions_out and numpy.all(enforce_periodic))

            state = self.__context.getState(getPositions=positions_out, getVelocities=velocities_out, getEnergy=energies_out, enforcePeriodicBox=enforce_periodic_flag)

        if vectors_out:
            state_results.append(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit_system(openmm.unit.md_unit_system))

        if positions_out:
            positions = state.getPositions(asNumpy=True).value_in_unit_system(openmm.unit.md_unit_system)

            # Determine whether or not an extra call to getState() needs to be
            # made to enforce periodic boundary conditions (the case if some
            # instances are requesting wrapped coordinates and others are not).
            get_wrapped_state_flag = bool(not enforce_periodic_flag and numpy.any(enforce_periodic))

            # See if we need to get wrapped coordinates in addition to unwrapped
            # coordinates already retrieved.
            if get_wrapped_state_flag:
                wrapped_state = self.__context.getState(getPositions=True, enforcePeriodicBox=True)
                wrapped_positions = wrapped_state.getPositions(asNumpy=True).value_in_unit_system(openmm.unit.md_unit_system)
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
            state_results.append(state.getVelocities(asNumpy=True).value_in_unit_system(openmm.unit.md_unit_system))

        if energies_out:
            state_results.append(state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system))
            state_results.append(state.getKineticEnergy().value_in_unit_system(openmm.unit.md_unit_system))

        return command_results, state_results

    def apply_constraints(self, positions, velocities):
        if positions:
            self.__context.applyConstraints(self.__integrator.getConstraintTolerance())
        if velocities:
            self.__context.applyVelocityConstraints(self.__integrator.getConstraintTolerance())

    def minimize(self, tolerance, iteration_count):
        openmm.LocalEnergyMinimizer.minimize(self.__context, tolerance, iteration_count)

    def maxwell_boltzmann(self, run_temperature):
        self.__context.setVelocitiesToTemperature(run_temperature, support.get_seed(self.__rng))

    def integrate(self, step_count, write_start, write_stop, write_step):
        step_index = 0
        write_pointer = None
        write_count = 0

        for write_index in range(step_count)[write_start:write_stop:write_step]:
            # See what step we are at, what step we need to get to to make the
            # next write, and how many steps we need to simulate to get there.
            integrate_count = write_index - step_index
            if integrate_count:
                self.__integrator.step(integrate_count)
            step_index += integrate_count

            # If this is the first write, make sure that we have a trajectory
            # file open and that we record the point in the trajectory file at
            # which the writing is starting.
            if not write_count:
                if self.__traj_file is None:
                    self.__traj_file = tempfile.NamedTemporaryFile(prefix="multiopenmm_traj_", suffix=".bin", dir=support.get_scratch_directory(), delete=False)
                write_pointer = self.__traj_file.tell()

            # Retrieve vectors and positions to write to the trajectory file.
            state = self.__context.getState(getPositions=True)
            vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit_system(openmm.unit.md_unit_system).astype(numpy.float64)
            positions = state.getPositions(asNumpy=True).value_in_unit_system(openmm.unit.md_unit_system).astype(numpy.float64)

            # Write the number of particles N as a uint64, followed by a 3-by-3
            # matrix of float64 values in C order containing periodic box vector
            # components, followed by an N-by-3 matrix of float64 values in C
            # order containing position coordinates.
            self.__traj_file.write(positions.shape[0].to_bytes(8, byteorder="little"))
            self.__traj_file.write(vectors.tobytes())
            self.__traj_file.write(positions.tobytes())

            write_count += 1

        # Make sure that, upon returning, the expected number of frames will
        # have been written completely to the trajectory file.
        if write_count:
            self.__traj_file.flush()

        # Finish if we have additional steps to simulate after the last write.
        integrate_count = step_count - step_index
        if integrate_count:
            self.__integrator.step(integrate_count)

        return self.__traj_file.name, write_pointer, write_count, self.__particle_offsets
