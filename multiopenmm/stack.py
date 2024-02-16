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
import itertools
import numpy
import openmm

def stack(system, temperature_scales):
    """
    Prepares an OpenMM system containing multiple non-interacting copies of a
    given OpenMM system.  Each copy will have its parameters adjusted so as to
    sample a configurational distribution with a temperature scaled by a
    particular amount from the temperature at which the combined system is
    simulated.

    Parameters
    ----------
    system : openmm.openmm.System
        An OpenMM system representing a single copy.
    temperature_scales : array of float
        The temperature scale factors.  The number of factors given will
        determine the number of copies created in the combined system.  A scale
        factor of 1 will result in the corresponding copy sampling a
        configurational distribution with a temperature equal to that at which
        the combined system is simulated.

    Returns
    -------
    openmm.openmm.System
        An OpenMM system containing multiple non-interacting copies of the given
        system.
    """

    if not isinstance(system, openmm.System):
        raise TypeError("system must be an OpenMM System")

    temperature_scales = numpy.atleast_1d(numpy.asarray(temperature_scales, dtype=float))
    if temperature_scales.ndim != 1:
        raise ValueError("temperature_scales must be 1-dimensional")

    particle_count = system.getNumParticles()
    constraint_count = system.getNumConstraints()
    copy_count = temperature_scales.size
    energy_scales = 1 / temperature_scales

    stacked_system = openmm.System()

    # Copy periodic box vectors.
    stacked_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    
    # Create particles.  Scale kinetic energies appropriately.
    for copy_index, energy_scale in enumerate(energy_scales):
        for particle_index in range(particle_count):
            stacked_system.addParticle(energy_scale * system.getParticleMass(particle_index))

    # Create constraints.  Offset particle indices appropriately.
    for copy_index in range(copy_count):
        offset = copy_index * particle_count

        for constraint_index in range(constraint_count):
            particle_index_1, particle_index_2, distance = system.getConstraintParameters(constraint_index)
            stacked_system.addConstraint(particle_index_1 + offset, particle_index_2 + offset, distance)

    # Create virtual sites.
    for copy_index in range(copy_count):
        offset = copy_index * particle_count

        for particle_index in range(particle_count):
            if system.isVirtualSite(particle_index):
                stacked_system.setVirtualSite(particle_index + offset, DefaultVirtualSiteProcessor._process(system.getVirtualSite(particle_index), offset))

    # Create forces.
    for force_index in range(system.getNumForces()):
        for force in DefaultForceProcessor._process(system.getForce(force_index), particle_count, energy_scales):
            stacked_system.addForce(force)

    return stacked_system

class Processor(abc.ABC):
    # Represents a generic processor that allows handlers for specific types of
    # objects that might be encountered in an OpenMM system to be registered.

    __slots__ = ("__handler_registry", "__handler_table")

    def __init__(self):
        self.__handler_registry = []
        self.__handler_table = None

    def register_handler(self, handler):
        self._check_handler(handler)

        # Reset the handler table when the handler registry is updated.  The
        # table will be regenerated at the next access of the _handler_table
        # property.
        self.__handler_registry.append(handler)
        self.__handler_table = None

    @property
    def _handler_table(self):
        # If the handler table is None, it was either never created or it was
        # reset because a new handler was registered.  Rebuild it such that
        # handlers added later override any mappings of types from handlers
        # added earlier.  In this way, custom handlers can be registered that
        # override the behavior of the default handlers.
        if self.__handler_table is None:
            self.__handler_table = {}
            for handler in self.__handler_registry:
                for handled_type in handler.handled_types:
                    self.__handler_table[handled_type] = handler

        return self.__handler_table

    @abc.abstractmethod
    def _check_handler(self, handler):
        # This method can be overridden in subclasses to perform, e.g., type
        # validation, or other checks, on handlers that are being registered.

        pass

    def _process(self, item, *args, **kwargs):
        handler_table = self._handler_table
        item_type = type(item)

        # When processing an item, search through its method resolution order
        # list such that the handler table will be searched first for the type
        # of the item, followed by base classes.
        for base_type in item_type.mro():
            if base_type in handler_table:
                return handler_table[base_type].handle(item, *args, **kwargs)
        else:
            raise TypeError(f"{item_type.__name__} is unsupported")

class VirtualSiteProcessor(Processor):
    """
    Represents a processor for dispatching the handling of
    :py:class:`openmm.openmm.VirtualSite` objects to registered handler objects.

    .. py:method:: register_handler(handler)

        Registers a virtual site handler.  All virtual site types that the
        handler reports as handleable will be mapped to the handler, overriding
        the mappings of any previous handlers also reporting one or more of the
        same virtual site types.

        :type handler: multiopenmm.stack.VirtualSiteHandler
        :param handler: The handler to register.
    """

    __slots__ = ()

    def _check_handler(self, handler):
        if not isinstance(handler, VirtualSiteHandler):
            raise TypeError("handler must be a VirtualSiteHandler")

class VirtualSiteHandler(abc.ABC):
    """
    An abstract class that can be inherited from to define a custom handler for
    one or more :py:class:`openmm.openmm.VirtualSite` subclasses.

    Once such a handler has been defined, an instance of it must be registered,
    *e.g.*:

    .. code-block::

        class MyCustomVirtualSiteHandler(multiopenmm.stack.VirtualSiteHandler):
            @property
            def handled_types(self):
                return (MyCustomOpenMMVirtualSite,)

            def handle(self, site, offset):
                ...
                return MyCustomOpenMMVirtualSite(...)

        multiopenmm.stack.DefaultVirtualSiteProcessor.register_handler(MyCustomVirtualSiteHandler())
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def handled_types(self):
        """
        tuple(type): The types of objects that can be handled by this handler.
        Each type should be a subclass of :py:class:`openmm.openmm.VirtualSite`.

        Notes
        -----
        This property must be implemented in derived classes.  A handler only
        handling objects of a single type should return a tuple of length 1.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def handle(self, site, offset):
        """
        Creates a copy of a virtual site for use in a stacked system.

        Parameters
        ----------
        site : openmm.openmm.VirtualSite
            The virtual site to create a copy of.
        offset : int
            The starting index of particles in a particular copy of an OpenMM
            system being created for the stacked system that this copy of the
            given virtual site is being created for.  In a system of :math:`N`
            particles being made into a stacked system containing :math:`M`
            copies of it, the offsets will be :math:`0,N,2N,\\ldots,(M-1)N`.
        
        Returns
        -------
        openmm.openmm.VirtualSite
            A new virtual site with the given offset applied to its particle
            indices, and otherwise identical to the given virtual site.

        Notes
        -----
        This method must be implemented in derived classes.  The virtual site
        returned need not be of the same type as the given virtual site.
        """

        raise NotImplementedError

class TwoParticleAverageSiteHandler(VirtualSiteHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.TwoParticleAverageSite,)

    def handle(self, site, offset):
        return openmm.TwoParticleAverageSite(
            site.getParticle(0) + offset,
            site.getParticle(1) + offset,
            site.getWeight(0),
            site.getWeight(1),
        )

class ThreeParticleAverageSiteHandler(VirtualSiteHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.ThreeParticleAverageSite,)

    def handle(self, site, offset):
        return openmm.ThreeParticleAverageSite(
            site.getParticle(0) + offset,
            site.getParticle(1) + offset,
            site.getParticle(2) + offset,
            site.getWeight(0),
            site.getWeight(1),
            site.getWeight(2),
        )

class OutOfPlaneSiteHandler(VirtualSiteHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.OutOfPlaneSite,)

    def handle(self, site, offset):
        return openmm.OutOfPlaneSite(
            site.getParticle(0) + offset,
            site.getParticle(1) + offset,
            site.getParticle(2) + offset,
            site.getWeight12(),
            site.getWeight13(),
            site.getWeightCross(),
        )

class LocalCoordinatesSiteHandler(VirtualSiteHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.LocalCoordinatesSite,)

    def handle(self, site, offset):
        return openmm.LocalCoordinatesSite(
            [site.getParticle(particle_index) + offset for particle_index in range(site.getNumParticles())],
            site.getOriginWeights(),
            site.getXWeights(),
            site.getYWeights(),
            site.getLocalPosition(),
        )

class ForceProcessor(Processor):
    """
    Represents a processor for dispatching the handling of
    :py:class:`openmm.openmm.Force` objects to registered handler objects.

    .. py:method:: register_handler(handler)

        Registers a force handler.  All force types that the handler reports as
        handleable will be mapped to the handler, overriding the mappings of any
        previous handlers also reporting one or more of the same force types.

        :type handler: multiopenmm.stack.ForceHandler
        :param handler: The handler to register.
    """

    __slots__ = ()

    def _check_handler(self, handler):
        if not isinstance(handler, ForceHandler):
            raise TypeError("handler must be a ForceHandler")

class ForceHandler(abc.ABC):
    """
    An abstract class that can be inherited from to define a custom handler for
    one or more :py:class:`openmm.openmm.Force` subclasses.

    Once such a handler has been defined, an instance of it must be registered,
    *e.g.*:

    .. code-block::

        class MyCustomForceHandler(multiopenmm.stack.ForceHandler):
            @property
            def handled_types(self):
                return (MyCustomOpenMMForce,)

            def _handle(self, force, particle_count, energy_scales):
                ...
                yield MyCustomOpenMMForce(...)

        multiopenmm.stack.DefaultForceProcessor.register_handler(MyCustomForceHandler())
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def handled_types(self):
        """
        tuple(type): The types of objects that can be handled by this handler.
        Each type should be a subclass of :py:class:`openmm.openmm.Force`.

        Notes
        -----
        This property must be implemented in derived classes.  A handler only
        handling objects of a single type should return a tuple of length 1.
        """

        raise NotImplementedError

    def handle(self, force, particle_count, energy_scales):
        # All Force objects have a Name and a ForceGroup that can be copied.
        # _handle() is deferred to for type-specific details.
        for stacked_force in self._handle(force, particle_count, energy_scales):
            stacked_force.setName(force.getName())
            stacked_force.setForceGroup(force.getForceGroup())
            yield stacked_force

    @abc.abstractmethod
    def _handle(self, force, particle_count, energy_scales):
        """
        Creates forces representing a copy of a force for use in a stacked
        system.

        :meta public:

        Parameters
        ----------
        force : openmm.openmm.Force
            The force to copy.
        particle_count : int
            The number of particles in the OpenMM system from which a stacked
            system is being created.
        energy_scales : array of float
            The factors by which interaction energies in each of the copies in
            the stacked system are to be scaled.

        Returns
        -------
        iterable(openmm.openmm.Force)
            A collection of forces to be added to the stacked system to
            replicate the behavior of the given force being applied to each of
            the non-interacting copies within the stacked system.

        Notes
        -----
        This method must be implemented in derived classes.  The forces returned
        need not be of the same type as the given force.  If multiple forces are
        unnecessary, an iterable yielding a single force should be returned.
        """

        raise NotImplementedError

class HarmonicBondForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.HarmonicBondForce,)

    def _handle(self, force, particle_count, energy_scales):
        bond_count = force.getNumBonds()

        bond_parameters = [force.getBondParameters(bond_index) for bond_index in range(bond_count)]

        stacked_force = openmm.HarmonicBondForce()
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for (
                    particle_index_1, particle_index_2,
                    length, k,
                ) in bond_parameters:

                stacked_force.addBond(
                    particle_index_1 + particle_offset,
                    particle_index_2 + particle_offset,
                    length,
                    energy_scale * k,
                )

        yield stacked_force

class HarmonicAngleForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.HarmonicAngleForce,)

    def _handle(self, force, particle_count, energy_scales):
        angle_count = force.getNumAngles()

        angle_parameters = [force.getAngleParameters(angle_index) for angle_index in range(angle_count)]

        stacked_force = openmm.HarmonicAngleForce()
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for (
                    particle_index_1, particle_index_2, particle_index_3,
                    angle, k,
                ) in angle_parameters:

                stacked_force.addAngle(
                    particle_index_1 + particle_offset,
                    particle_index_2 + particle_offset,
                    particle_index_3 + particle_offset,
                    angle,
                    energy_scale * k,
                )

        yield stacked_force

class PeriodicTorsionForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.PeriodicTorsionForce,)

    def _handle(self, force, particle_count, energy_scales):
        torsion_count = force.getNumTorsions()

        torsion_parameters = [force.getTorsionParameters(torsion_index) for torsion_index in range(torsion_count)]

        stacked_force = openmm.PeriodicTorsionForce()
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for (
                    particle_index_1, particle_index_2, particle_index_3, particle_index_4,
                    periodicity, phase, k,
                ) in torsion_parameters:

                stacked_force.addTorsion(
                    particle_index_1 + particle_offset,
                    particle_index_2 + particle_offset,
                    particle_index_3 + particle_offset,
                    particle_index_4 + particle_offset,
                    periodicity,
                    phase,
                    energy_scale * k,
                )

        yield stacked_force

class RBTorsionForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.RBTorsionForce,)

    def _handle(self, force, particle_count, energy_scales):
        torsion_count = force.getNumTorsions()

        torsion_parameters = [force.getTorsionParameters(torsion_index) for torsion_index in range(torsion_count)]

        stacked_force = openmm.RBTorsionForce()
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for (
                    particle_index_1, particle_index_2, particle_index_3, particle_index_4,
                    c_0, c_1, c_2, c_3, c_4, c_5,
                ) in torsion_parameters:

                stacked_force.addTorsion(
                    particle_index_1 + particle_offset,
                    particle_index_2 + particle_offset,
                    particle_index_3 + particle_offset,
                    particle_index_4 + particle_offset,
                    energy_scale * c_0,
                    energy_scale * c_1,
                    energy_scale * c_2,
                    energy_scale * c_3,
                    energy_scale * c_4,
                    energy_scale * c_5,
                )

        yield stacked_force

class CMAPTorsionForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CMAPTorsionForce,)

    def _handle(self, force, particle_count, energy_scales):
        map_count = force.getNumMaps()
        torsion_count = force.getNumTorsions()

        map_parameters = [force.getMapParameters(map_index) for map_index in range(map_count)]
        torsion_parameters = [force.getTorsionParameters(torsion_index) for torsion_index in range(torsion_count)]

        stacked_force = openmm.CMAPTorsionForce()
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        for copy_index, energy_scale in enumerate(energy_scales):
            map_offset = copy_index * map_count
            particle_offset = copy_index * particle_count

            for size, energy in map_parameters:
                stacked_force.addMap(size, [energy_scale * energy_ij for energy_ij in energy])

            for (
                    map_index,
                    particle_index_a_1, particle_index_a_2, particle_index_a_3, particle_index_a_4,
                    particle_index_b_1, particle_index_b_2, particle_index_b_3, particle_index_b_4,
                ) in torsion_parameters:

                stacked_force.addTorsion(
                    map_index + map_offset,
                    particle_index_a_1 + particle_offset,
                    particle_index_a_2 + particle_offset,
                    particle_index_a_3 + particle_offset,
                    particle_index_a_4 + particle_offset,
                    particle_index_b_1 + particle_offset,
                    particle_index_b_2 + particle_offset,
                    particle_index_b_3 + particle_offset,
                    particle_index_b_4 + particle_offset,
                )

        yield stacked_force

class CustomExternalForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomExternalForce,)

    def _handle(self, force, particle_count, energy_scales):
        term_count = force.getNumParticles()
        term_parameter_count = force.getNumPerParticleParameters()
        global_parameter_count = force.getNumGlobalParameters()

        term_parameters = [force.getParticleParameters(term_index) for term_index in range(term_count)]
        term_parameter_names = [force.getPerParticleParameterName(term_parameter_index) for term_parameter_index in range(term_parameter_count)]
        global_parameter_names = [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        global_parameter_values = [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(global_parameter_count)]

        energy_function = force.getEnergyFunction()
        scale_name = _get_scale_name(energy_function, term_parameter_names + global_parameter_names)
        stacked_force = openmm.CustomExternalForce(_scale_function(energy_function, scale_name))

        stacked_force.addPerParticleParameter(scale_name)
        for term_parameter_name in term_parameter_names:
            stacked_force.addPerParticleParameter(term_parameter_name)

        for global_parameter_name, global_parameter_value in zip(global_parameter_names, global_parameter_values):
            stacked_force.addGlobalParameter(global_parameter_name, global_parameter_value)

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for particle_index, term_parameter_values in term_parameters:

                stacked_force.addParticle(
                    particle_index + particle_offset,
                    [energy_scale, *term_parameter_values],
                )
        
        yield stacked_force

class CustomBondForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomBondForce,)

    def _handle(self, force, particle_count, energy_scales):
        bond_count = force.getNumBonds()
        bond_parameter_count = force.getNumPerBondParameters()
        global_parameter_count = force.getNumGlobalParameters()
        derivative_count = force.getNumEnergyParameterDerivatives()

        bond_parameters = [force.getBondParameters(bond_index) for bond_index in range(bond_count)]
        bond_parameter_names = [force.getPerBondParameterName(bond_parameter_index) for bond_parameter_index in range(bond_parameter_count)]
        global_parameter_names = [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        global_parameter_values = [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        derivative_names = [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(derivative_count)]

        energy_function = force.getEnergyFunction()
        scale_name = _get_scale_name(energy_function, bond_parameter_names + global_parameter_names + derivative_names)
        stacked_force = openmm.CustomBondForce(_scale_function(energy_function, scale_name))
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        stacked_force.addPerBondParameter(scale_name)
        for bond_parameter_name in bond_parameter_names:
            stacked_force.addPerBondParameter(bond_parameter_name)

        for global_parameter_name, global_parameter_value in zip(global_parameter_names, global_parameter_values):
            stacked_force.addGlobalParameter(global_parameter_name, global_parameter_value)

        for derivative_name in derivative_names:
            stacked_force.addEnergyParameterDerivative(derivative_name)

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for (
                    particle_index_1, particle_index_2,
                    bond_parameter_values,
                ) in bond_parameters:

                stacked_force.addBond(
                    particle_index_1 + particle_offset,
                    particle_index_2 + particle_offset,
                    [energy_scale, *bond_parameter_values],
                )

        yield stacked_force

class CustomAngleForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomAngleForce,)

    def _handle(self, force, particle_count, energy_scales):
        angle_count = force.getNumAngles()
        angle_parameter_count = force.getNumPerAngleParameters()
        global_parameter_count = force.getNumGlobalParameters()
        derivative_count = force.getNumEnergyParameterDerivatives()

        angle_parameters = [force.getAngleParameters(angle_index) for angle_index in range(angle_count)]
        angle_parameter_names = [force.getPerAngleParameterName(angle_parameter_index) for angle_parameter_index in range(angle_parameter_count)]
        global_parameter_names = [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        global_parameter_values = [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        derivative_names = [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(derivative_count)]

        energy_function = force.getEnergyFunction()
        scale_name = _get_scale_name(energy_function, angle_parameter_names + global_parameter_names + derivative_names)
        stacked_force = openmm.CustomAngleForce(_scale_function(energy_function, scale_name))
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        stacked_force.addPerAngleParameter(scale_name)
        for angle_parameter_name in angle_parameter_names:
            stacked_force.addPerAngleParameter(angle_parameter_name)

        for global_parameter_name, global_parameter_value in zip(global_parameter_names, global_parameter_values):
            stacked_force.addGlobalParameter(global_parameter_name, global_parameter_value)

        for derivative_name in derivative_names:
            stacked_force.addEnergyParameterDerivative(derivative_name)

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for (
                    particle_index_1, particle_index_2, particle_index_3,
                    angle_parameter_values,
                ) in angle_parameters:

                stacked_force.addAngle(
                    particle_index_1 + particle_offset,
                    particle_index_2 + particle_offset,
                    particle_index_3 + particle_offset,
                    [energy_scale, *angle_parameter_values],
                )

        yield stacked_force

class CustomTorsionForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomTorsionForce,)

    def _handle(self, force, particle_count, energy_scales):
        torsion_count = force.getNumTorsions()
        torsion_parameter_count = force.getNumPerTorsionParameters()
        global_parameter_count = force.getNumGlobalParameters()
        derivative_count = force.getNumEnergyParameterDerivatives()

        torsion_parameters = [force.getTorsionParameters(torsion_index) for torsion_index in range(torsion_count)]
        torsion_parameter_names = [force.getPerTorsionParameterName(torsion_parameter_index) for torsion_parameter_index in range(torsion_parameter_count)]
        global_parameter_names = [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        global_parameter_values = [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        derivative_names = [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(derivative_count)]

        energy_function = force.getEnergyFunction()
        scale_name = _get_scale_name(energy_function, torsion_parameter_names + global_parameter_names + derivative_names)
        stacked_force = openmm.CustomTorsionForce(_scale_function(energy_function, scale_name))
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        stacked_force.addPerTorsionParameter(scale_name)
        for torsion_parameter_name in torsion_parameter_names:
            stacked_force.addPerTorsionParameter(torsion_parameter_name)

        for global_parameter_name, global_parameter_value in zip(global_parameter_names, global_parameter_values):
            stacked_force.addGlobalParameter(global_parameter_name, global_parameter_value)

        for derivative_name in derivative_names:
            stacked_force.addEnergyParameterDerivative(derivative_name)

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for (
                    particle_index_1, particle_index_2, particle_index_3, particle_index_4,
                    torsion_parameter_values,
                ) in torsion_parameters:

                stacked_force.addTorsion(
                    particle_index_1 + particle_offset,
                    particle_index_2 + particle_offset,
                    particle_index_3 + particle_offset,
                    particle_index_4 + particle_offset,
                    [energy_scale, *torsion_parameter_values],
                )

        yield stacked_force

class CustomCompoundBondForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomCompoundBondForce,)

    def _handle(self, force, particle_count, energy_scales):
        bond_count = force.getNumBonds()
        bond_parameter_count = force.getNumPerBondParameters()
        global_parameter_count = force.getNumGlobalParameters()
        derivative_count = force.getNumEnergyParameterDerivatives()
        function_count = force.getNumTabulatedFunctions()

        bond_parameters = [force.getBondParameters(bond_index) for bond_index in range(bond_count)]
        bond_parameter_names = [force.getPerBondParameterName(bond_parameter_index) for bond_parameter_index in range(bond_parameter_count)]
        global_parameter_names = [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        global_parameter_values = [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        derivative_names = [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(derivative_count)]
        function_names = [force.getTabulatedFunctionName(function_index) for function_index in range(function_count)]
        functions = [force.getTabulatedFunction(function_index) for function_index in range(function_count)]

        energy_function = force.getEnergyFunction()
        scale_name = _get_scale_name(energy_function, bond_parameter_names + global_parameter_names + derivative_names + function_names)
        stacked_force = openmm.CustomCompoundBondForce(force.getNumParticlesPerBond(), _scale_function(energy_function, scale_name))
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        stacked_force.addPerBondParameter(scale_name)
        for bond_parameter_name in bond_parameter_names:
            stacked_force.addPerBondParameter(bond_parameter_name)

        for global_parameter_name, global_parameter_value in zip(global_parameter_names, global_parameter_values):
            stacked_force.addGlobalParameter(global_parameter_name, global_parameter_value)

        for derivative_name in derivative_names:
            stacked_force.addEnergyParameterDerivative(derivative_name)
        
        for function_name, function in zip(function_names, functions):
            stacked_force.addTabulatedFunction(function_name, openmm.XmlSerializer.clone(function))

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for particle_indices, bond_parameter_values in bond_parameters:

                stacked_force.addBond(
                    [particle_index + particle_offset for particle_index in particle_indices],
                    [energy_scale, *bond_parameter_values],
                )

        yield stacked_force

class CustomCentroidBondForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomCentroidBondForce,)

    def _handle(self, force, particle_count, energy_scales):
        group_count = force.getNumGroups()
        bond_count = force.getNumBonds()
        bond_parameter_count = force.getNumPerBondParameters()
        global_parameter_count = force.getNumGlobalParameters()
        derivative_count = force.getNumEnergyParameterDerivatives()
        function_count = force.getNumTabulatedFunctions()

        group_parameters = [force.getGroupParameters(group_index) for group_index in range(group_count)]
        bond_parameters = [force.getBondParameters(bond_index) for bond_index in range(bond_count)]
        bond_parameter_names = [force.getPerBondParameterName(bond_parameter_index) for bond_parameter_index in range(bond_parameter_count)]
        global_parameter_names = [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        global_parameter_values = [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        derivative_names = [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(derivative_count)]
        function_names = [force.getTabulatedFunctionName(function_index) for function_index in range(function_count)]
        functions = [force.getTabulatedFunction(function_index) for function_index in range(function_count)]

        energy_function = force.getEnergyFunction()
        scale_name = _get_scale_name(energy_function, bond_parameter_names + global_parameter_names + derivative_names + function_names)
        stacked_force = openmm.CustomCentroidBondForce(force.getNumGroupsPerBond(), _scale_function(energy_function, scale_name))
        stacked_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())

        stacked_force.addPerBondParameter(scale_name)
        for bond_parameter_name in bond_parameter_names:
            stacked_force.addPerBondParameter(bond_parameter_name)

        for global_parameter_name, global_parameter_value in zip(global_parameter_names, global_parameter_values):
            stacked_force.addGlobalParameter(global_parameter_name, global_parameter_value)

        for derivative_name in derivative_names:
            stacked_force.addEnergyParameterDerivative(derivative_name)
        
        for function_name, function in zip(function_names, functions):
            stacked_force.addTabulatedFunction(function_name, openmm.XmlSerializer.clone(function))

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for particle_indices, weights in group_parameters:
                
                stacked_force.addGroup(
                    [particle_index + particle_offset for particle_index in particle_indices],
                    weights,
                )

        for copy_index, energy_scale in enumerate(energy_scales):
            group_offset = copy_index * group_count

            for group_indices, bond_parameter_values in bond_parameters:

                stacked_force.addBond(
                    [group_index + group_offset for group_index in group_indices],
                    [energy_scale, *bond_parameter_values],
                )

        yield stacked_force

class CustomNonbondedForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomNonbondedForce,)

    def _handle(self, force, particle_count, energy_scales):
        exclusion_count = force.getNumExclusions()
        group_count = force.getNumInteractionGroups()
        compute_count = force.getNumComputedValues()
        particle_parameter_count = force.getNumPerParticleParameters()
        global_parameter_count = force.getNumGlobalParameters()
        derivative_count = force.getNumEnergyParameterDerivatives()
        function_count = force.getNumTabulatedFunctions()

        exclusion_parameters = [force.getExclusionParticles(exclusion_index) for exclusion_index in range(exclusion_count)]
        group_parameters = [force.getInteractionGroupParameters(group_index) for group_index in range(group_count)]
        compute_parameters = [force.getComputedValueParameters(compute_index) for compute_index in range(compute_count)]
        particle_parameters = [force.getParticleParameters(particle_index) for particle_index in range(particle_count)]
        particle_parameter_names = [force.getPerParticleParameterName(particle_parameter_index) for particle_parameter_index in range(particle_parameter_count)]
        global_parameter_names = [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        global_parameter_values = [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(global_parameter_count)]
        derivative_names = [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(derivative_count)]
        function_names = [force.getTabulatedFunctionName(function_index) for function_index in range(function_count)]
        functions = [force.getTabulatedFunction(function_index) for function_index in range(function_count)]

        # If no interaction groups are present, add one containing all
        # particles.  This will give the same behavior if a single copy of the
        # system is being made, and will ensure that multiple copies will not
        # interact with each other even if interaction groups are initially
        # absent.
        if not group_count:
            group_count = 1
            group_parameters.append([tuple(range(particle_count))] * 2)

        # In the energy function, retrieve the energy scale from the first of
        # the two particles in an interacting pair.  Since only particles from
        # the same copies of the system will ever interact, the energy scale
        # should have the same value for both particles in any interacting pair.
        energy_function = force.getEnergyFunction()
        scale_name = _get_scale_name(energy_function, [compute_name for compute_name, compute in compute_parameters] + particle_parameter_names + global_parameter_names + derivative_names + function_names)
        stacked_force = openmm.CustomNonbondedForce(_scale_function(energy_function, f"{scale_name}1"))
        stacked_force.setNonbondedMethod(force.getNonbondedMethod())
        stacked_force.setUseSwitchingFunction(force.getUseSwitchingFunction())
        stacked_force.setUseLongRangeCorrection(force.getUseLongRangeCorrection())
        stacked_force.setCutoffDistance(force.getCutoffDistance())
        stacked_force.setSwitchingDistance(force.getSwitchingDistance())

        for compute_name, compute in compute_parameters:
            stacked_force.addComputedValue(compute_name, compute)

        stacked_force.addPerParticleParameter(scale_name)
        for particle_parameter_name in particle_parameter_names:
            stacked_force.addPerParticleParameter(particle_parameter_name)

        for global_parameter_name, global_parameter_value in zip(global_parameter_names, global_parameter_values):
            stacked_force.addGlobalParameter(global_parameter_name, global_parameter_value)

        for derivative_name in derivative_names:
            stacked_force.addEnergyParameterDerivative(derivative_name)
        
        for function_name, function in zip(function_names, functions):
            stacked_force.addTabulatedFunction(function_name, openmm.XmlSerializer.clone(function))

        for copy_index, energy_scale in enumerate(energy_scales):
            particle_offset = copy_index * particle_count

            for particle_parameter_values in particle_parameters:
                stacked_force.addParticle([energy_scale, *particle_parameter_values])

            for particle_index_1, particle_index_2 in exclusion_parameters:
                stacked_force.addExclusion(
                    particle_index_1 + particle_offset,
                    particle_index_2 + particle_offset,
                )

            for particle_indices_1, particle_indices_2 in group_parameters:
                stacked_force.addInteractionGroup(
                    [particle_index_1 + particle_offset for particle_index_1 in particle_indices_1],
                    [particle_index_2 + particle_offset for particle_index_2 in particle_indices_2],
                )

        yield stacked_force

def _get_scale_name(function, parameter_names, prefix="scale"):
    parameter_names = set(parameter_names)
    for length in itertools.count():
        for suffix in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=length):
            name = prefix + "".join(map(str, suffix))
            if name not in parameter_names and name not in function:
                return name

def _scale_function(function, scale_name):
    expression, *definitions = function.split(";")
    return ";".join([f"{scale_name}*({expression})", *definitions])

#: multiopenmm.stack.VirtualSiteProcessor: The virtual site processor used by
#: :py:func:`multiopenmm.stack.stack` to process all virtual sites in OpenMM
#: systems.
DefaultVirtualSiteProcessor = VirtualSiteProcessor()

DefaultVirtualSiteProcessor.register_handler(TwoParticleAverageSiteHandler())
DefaultVirtualSiteProcessor.register_handler(ThreeParticleAverageSiteHandler())
DefaultVirtualSiteProcessor.register_handler(OutOfPlaneSiteHandler())
DefaultVirtualSiteProcessor.register_handler(LocalCoordinatesSiteHandler())

#: multiopenmm.stack.ForceProcessor: The force processor used by
#: :py:func:`multiopenmm.stack.stack` to process all forces in OpenMM systems.
DefaultForceProcessor = ForceProcessor()

DefaultForceProcessor.register_handler(HarmonicBondForceHandler())
DefaultForceProcessor.register_handler(HarmonicAngleForceHandler())
DefaultForceProcessor.register_handler(PeriodicTorsionForceHandler())
DefaultForceProcessor.register_handler(RBTorsionForceHandler())
DefaultForceProcessor.register_handler(CMAPTorsionForceHandler())
DefaultForceProcessor.register_handler(CustomExternalForceHandler())
DefaultForceProcessor.register_handler(CustomBondForceHandler())
DefaultForceProcessor.register_handler(CustomAngleForceHandler())
DefaultForceProcessor.register_handler(CustomTorsionForceHandler())
DefaultForceProcessor.register_handler(CustomCompoundBondForceHandler())
DefaultForceProcessor.register_handler(CustomCentroidBondForceHandler())
DefaultForceProcessor.register_handler(CustomNonbondedForceHandler())
