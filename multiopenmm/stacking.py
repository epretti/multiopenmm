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
import warnings

from . import support

def stack(templates, template_indices, temperature_scales):
    """
    Prepares an OpenMM system containing multiple non-interacting molecular
    system instances, templated from one or more given OpenMM systems.  Each
    instance will have its parameters adjusted so as to sample a configurational
    distribution with a temperature scaled by a specified amount from the
    temperature at which the combined system is simulated.

    Parameters
    ----------
    templates : iterable of openmm.openmm.System
        OpenMM systems serving as templates for molecular system instances.
    template_indices : array of int
        Indices specifying which template OpenMM system each instance should be
        an instance of.  The number of indices given will determine the number
        of instances created in the combined system.
    temperature_scales : array of float
        The temperature scale factors.  The number of factors given must match
        the number of template indices given.  A scale factor of 1 will result
        in the corresponding instance sampling a configurational distribution
        with a temperature equal to that at which the combined system is
        simulated.

    Returns
    -------
    (openmm.openmm.System, array of int)
        An OpenMM system containing multiple non-interacting molecular system
        instances, templated after the specified OpenMM systems, and scaled as
        specified.  Also, for each instance, the particle index in the combined
        system corresponding to the first particle in the instance, followed by
        the total number of particles in the combined system (such that the
        differences of consecutive offsets give the numbers of particles in each
        instance).  If the combined system has :math:`M` molecular simulation
        instances containing :math:`N_1`, :math:`N_2`, :math:`\\ldots`,
        :math:`N_M` particles in turn, the offsets will be :math:`0`,
        :math:`N_1`, :math:`N_1+N_2`, :math:`\\ldots`,
        :math:`\\sum_{i=1}^{M-1}N_i`, :math:`\\sum_{i=1}^MN_i`.

    Notes
    -----
    If there are multiple templates in use with default periodic box vectors
    that differ from one another, the default periodic box vectors from the
    first template in use will be selected for use in the combined system.  A
    warning will be issued if this condition is detected; in general, it
    indicates a problem with the desired combined system, as all instances
    within must share the same periodic boundaries.
    """

    # Check all system objects.
    template_list = []
    for template in templates:
        if not isinstance(template, openmm.System):
            raise TypeError("template must be an OpenMM System")
        template_list.append(template)

    all_template_count = len(template_list)

    # Check template indices.
    template_indices = numpy.atleast_1d(numpy.asarray(template_indices, dtype=int))
    if template_indices.ndim != 1:
        raise ValueError("template_indices must be 1-dimensional")

    if numpy.any((template_indices < 0) | (template_indices >= all_template_count)):
        # If no templates were given, no valid template indices will exist, so
        # generate an appropriate error message in this case.
        raise ValueError(f"template indices must be non-negative and less than {all_template_count}"
            if all_template_count else "no templates given")

    instance_count = template_indices.size

    # Check temperature scales.
    temperature_scales = numpy.atleast_1d(numpy.asarray(temperature_scales, dtype=float))
    if temperature_scales.ndim != 1:
        raise ValueError("temperature_scales must be 1-dimensional")
    if temperature_scales.size != instance_count:
        raise ValueError("template_indices and temperature_scales must have the same shape")

    energy_scales = 1 / temperature_scales

    # Normalize templates and their indices to remove templates not used.
    template_indices_used, template_indices = numpy.unique(template_indices, return_inverse=True)
    templates = [template_list[template_index] for template_index in template_indices_used]

    # Retrieve template properties.
    particle_counts = [template.getNumParticles() for template in templates]
    particle_masses = [
        [template.getParticleMass(particle_index) for particle_index in range(particle_count)]
        for template, particle_count in zip(templates, particle_counts)
    ]
    virtual_sites = [
        [template.getVirtualSite(particle_index) if template.isVirtualSite(particle_index) else None for particle_index in range(particle_count)]
        for template, particle_count in zip(templates, particle_counts)
    ]
    constraint_counts = [template.getNumConstraints() for template in templates]
    constraint_parameters = [
        [template.getConstraintParameters(constraint_index) for constraint_index in range(constraint_count)]
        for template, constraint_count in zip(templates, constraint_counts)
    ]

    # Determine the number of particles in each instance and the offset of each
    # instance in the collection of all of the particles in the combined system.
    # Keep an additional index giving the total number of particles in the
    # combined system such that the number of particles in each instance can be
    # calculated as the differences of each pair of consecutive offsets.
    particle_offsets = numpy.concatenate(((0,), numpy.cumsum([particle_counts[template_index] for template_index in template_indices], dtype=int)))

    combined_system = openmm.System()

    # Set periodic box vectors.
    if templates:
        vectors = templates[0].getDefaultPeriodicBoxVectors()
        if any(template.getDefaultPeriodicBoxVectors() != vectors for template in templates[1:]):
            warnings.warn("Mismatched default periodic box vectors; choosing vectors from first template in use", support.MultiOpenMMWarning)
        combined_system.setDefaultPeriodicBoxVectors(*vectors)
    else:
        warnings.warn("No templates in use; default periodic box vectors will have default values", support.MultiOpenMMWarning)

    # Create particles.  Scale kinetic energies appropriately.
    for template_index, energy_scale in zip(template_indices, energy_scales):
        for particle_index in range(particle_counts[template_index]):
            combined_system.addParticle(energy_scale * particle_masses[template_index][particle_index])

    # Create constraints.  Offset particle indices appropriately.
    for template_index, particle_offset in zip(template_indices, particle_offsets):
        for constraint_index in range(constraint_counts[template_index]):
            particle_index_1, particle_index_2, distance = constraint_parameters[template_index][constraint_index]
            combined_system.addConstraint(particle_index_1 + particle_offset, particle_index_2 + particle_offset, distance)

    # Create virtual sites.
    for template_index, particle_offset in zip(template_indices, particle_offsets):
        for particle_index in range(particle_counts[template_index]):
            virtual_site = virtual_sites[template_index][particle_index]
            if virtual_site is not None:
                combined_system.setVirtualSite(particle_index + particle_offset, DefaultVirtualSiteProcessor._process(virtual_site, particle_offset))

    # Create forces.
    for force in DefaultForceProcessor._process(templates, template_indices, particle_offsets, energy_scales):
        combined_system.addForce(force)

    return combined_system, particle_offsets

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

        raise NotImplementedError

    @abc.abstractmethod
    def _process(self, *args, **kwargs):
        # This method can be overridden in subclasses to handle processing of,
        # e.g., single items or entire sets of items.

        raise NotImplementedError

class VirtualSiteProcessor(Processor):
    """
    Represents a processor for dispatching the handling of
    :py:class:`openmm.openmm.VirtualSite` objects to registered handler objects.

    .. py:method:: register_handler(handler)

        Registers a virtual site handler.  All virtual site types that the
        handler reports as handleable will be mapped to the handler, overriding
        the mappings of any previously registered handlers also reporting one or
        more of the same virtual site types.

        :type handler: multiopenmm.stacking.VirtualSiteHandler
        :param handler: The handler to register.
    """

    __slots__ = ()

    def _check_handler(self, handler):
        if not isinstance(handler, VirtualSiteHandler):
            raise TypeError("handler must be a VirtualSiteHandler")

    def _process(self, site, particle_offset):
        handler_table = self._handler_table
        site_type = type(site)

        # When processing a virtual site, search through its method resolution
        # order list such that the handler table will be searched first for the
        # type of the virtual site itself, followed by base classes.
        for base_type in site_type.mro():
            if base_type in handler_table:
                return handler_table[base_type].handle(site, particle_offset)
        else:
            raise TypeError(f"{site_type.__name__} is unsupported")

class VirtualSiteHandler(abc.ABC):
    """
    An abstract class that can be inherited from to define a custom handler for
    one or more :py:class:`openmm.openmm.VirtualSite` subclasses.

    Once such a handler has been defined, an instance of it must be registered,
    *e.g.*:

    .. code-block::

        class MyCustomVirtualSiteHandler(multiopenmm.stacking.VirtualSiteHandler):
            @property
            def handled_types(self):
                return (MyCustomOpenMMVirtualSite,)

            def handle(self, site, particle_offset):
                ...
                return MyCustomOpenMMVirtualSite(...)

        multiopenmm.stacking.DefaultVirtualSiteProcessor.register_handler(MyCustomVirtualSiteHandler())
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
    def handle(self, site, particle_offset):
        """
        Creates a copy of a virtual site for use in a combined system.

        Parameters
        ----------
        site : openmm.openmm.VirtualSite
            The virtual site to create a copy of.
        particle_offset : int
            The starting index of particles in a particular molecular system
            instance being created for the combined system that this copy of the
            given virtual site is being created for.  If the combined system has
            :math:`M` molecular simulation instances containing :math:`N_1`,
            :math:`N_2`, :math:`\\ldots`, :math:`N_M` particles in turn, the
            offsets will be, respectively, :math:`0`, :math:`N_1`,
            :math:`N_1+N_2`, :math:`\\ldots`, :math:`\\sum_{i=1}^{M-2}N_i`,
            :math:`\\sum_{i=1}^{M-1}N_i`.
        
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

    def handle(self, site, particle_offset):
        return openmm.TwoParticleAverageSite(
            site.getParticle(0) + particle_offset,
            site.getParticle(1) + particle_offset,
            site.getWeight(0),
            site.getWeight(1),
        )

class ThreeParticleAverageSiteHandler(VirtualSiteHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.ThreeParticleAverageSite,)

    def handle(self, site, particle_offset):
        return openmm.ThreeParticleAverageSite(
            site.getParticle(0) + particle_offset,
            site.getParticle(1) + particle_offset,
            site.getParticle(2) + particle_offset,
            site.getWeight(0),
            site.getWeight(1),
            site.getWeight(2),
        )

class OutOfPlaneSiteHandler(VirtualSiteHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.OutOfPlaneSite,)

    def handle(self, site, particle_offset):
        return openmm.OutOfPlaneSite(
            site.getParticle(0) + particle_offset,
            site.getParticle(1) + particle_offset,
            site.getParticle(2) + particle_offset,
            site.getWeight12(),
            site.getWeight13(),
            site.getWeightCross(),
        )

class LocalCoordinatesSiteHandler(VirtualSiteHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.LocalCoordinatesSite,)

    def handle(self, site, particle_offset):
        return openmm.LocalCoordinatesSite(
            [site.getParticle(particle_index) + particle_offset for particle_index in range(site.getNumParticles())],
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
        previously registered handlers also reporting one or more of the same
        force types.

        :type handler: multiopenmm.stacking.ForceHandler
        :param handler: The handler to register.
    """

    __slots__ = ()

    def _check_handler(self, handler):
        if not isinstance(handler, ForceHandler):
            raise TypeError("handler must be a ForceHandler")

    def _process(self, templates, template_indices, particle_offsets, energy_scales):
        handler_table = self._handler_table

        # Create tables of forces to be passed to each force handler that can
        # accept them.
        force_tables = {}
        for template_index, template in enumerate(templates):
            for force_index in range(template.getNumForces()):
                force = template.getForce(force_index)
                force_type = type(force)

                # When processing a force, search through its method resolution
                # order list such that the force table will be searched first
                # for the type of the force itself, followed by base classes.
                for base_type in force_type.mro():
                    if base_type in handler_table:
                        # For each force handler, group forces by force group
                        # (forces from each template in each force group will be
                        # handled separately) and then by template index.
                        force_tables.setdefault(handler_table[base_type], {}).setdefault(force.getForceGroup(), {}).setdefault(template_index, []).append(force)
                        break
                else:
                    raise TypeError(f"{force_type.__name__} is unsupported")

        # Call each handler with all of the forces collected for it.
        for handler, force_table in force_tables.items():
            # Each value in the force table is constructed as a dictionary
            # mapping from a force group index to a dictionary mapping from a
            # template index to a list of forces.  Process each group,
            # converting its dictionary to a nested list and noting that if no
            # forces matching a particular handler were found in a given
            # template for the given force group, no key will exist for the
            # template.
            handler_name = "+".join(handled_type.__name__ for handled_type in handler.handled_types)
            for force_group, group_force_table in force_table.items():
                group_force_table_array = [group_force_table.get(template_index, []) for template_index in range(len(templates))]
                for force_index, force in enumerate(handler.handle(group_force_table_array, template_indices, particle_offsets, energy_scales)):
                    force.setForceGroup(force_group)
                    force.setName(f"{handler_name}:{force_group}:{force_index}")
                    yield force

class ForceHandler(abc.ABC):
    """
    An abstract class that can be inherited from to define a custom handler for
    one or more :py:class:`openmm.openmm.Force` subclasses.

    Once such a handler has been defined, an instance of it must be registered,
    *e.g.*:

    .. code-block::

        class MyCustomForceHandler(multiopenmm.stacking.ForceHandler):
            @property
            def handled_types(self):
                return (MyCustomOpenMMForce,)

            def handle(self, force_table, template_indices, particle_offsets, energy_scales):
                ...
                yield MyCustomOpenMMForce(...)

        multiopenmm.stacking.DefaultForceProcessor.register_handler(MyCustomForceHandler())
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

    @abc.abstractmethod
    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        """
        Creates forces representing combinations of other forces for use in a
        combined system.

        Parameters
        ----------
        force_table : list of list of openmm.openmm.Force
            For each template OpenMM system, a list of forces to be handled.
        template_indices : array of int
            For each molecular system instance contained within the combined
            OpenMM system within which the created forces will be contained, the
            index of the template OpenMM system upon which it should be based.
        particle_offsets : array of int
            For each instance, the particle index in the combined system
            corresponding to the first particle in the instance, followed by the
            total number of particles in the combined system (such that the
            differences of consecutive offsets give the numbers of particles in
            each instance).  If the combined system has :math:`M` molecular
            simulation instances containing :math:`N_1`, :math:`N_2`,
            :math:`\\ldots`, :math:`N_M` particles in turn, the offsets will be
            :math:`0`, :math:`N_1`, :math:`N_1+N_2`, :math:`\\ldots`,
            :math:`\\sum_{i=1}^{M-1}N_i`, :math:`\\sum_{i=1}^MN_i`.
        energy_scales : array of float
            For each instance, the factor by which its interaction energy should
            be scaled.

        Returns
        -------
        iterable of openmm.openmm.Force
            A collection of forces to be added to the combined system to
            replicate the behavior of the given forces being applied to the
            specified instances.

        Notes
        -----
        This method must be implemented in derived classes.  The forces created
        need not be of the same types as the given forces.  If multiple forces
        are not needed, an iterable yielding a single force should be returned.
        Implementations need not set the force group or name of any of the
        forces yielded as these will be set automatically.
        """

        raise NotImplementedError

class HarmonicBondForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.HarmonicBondForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve bond parameters for all forces.
        bond_parameters = [
            [
                [force.getBondParameters(bond_index) for bond_index in range(force.getNumBonds())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]
        
        # Split forces into classes based on whether or not they use periodic
        # boundary conditions.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault(force.usesPeriodicBoundaryConditions(), {}).setdefault(template_index, []).append(force_index)

        for uses_pbc, class_forces in force_classes.items():
            combined_force = openmm.HarmonicBondForce()
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for (
                            particle_index_1, particle_index_2,
                            length, k,
                        ) in bond_parameters[template_index][force_index]:

                        combined_force.addBond(
                            particle_index_1 + particle_offset,
                            particle_index_2 + particle_offset,
                            length,
                            energy_scale * k,
                        )

            yield combined_force

class HarmonicAngleForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.HarmonicAngleForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve angle parameters for all forces.
        angle_parameters = [
            [
                [force.getAngleParameters(angle_index) for angle_index in range(force.getNumAngles())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]
        
        # Split forces into classes based on whether or not they use periodic
        # boundary conditions.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault(force.usesPeriodicBoundaryConditions(), {}).setdefault(template_index, []).append(force_index)

        for uses_pbc, class_forces in force_classes.items():
            combined_force = openmm.HarmonicAngleForce()
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for (
                            particle_index_1, particle_index_2, particle_index_3,
                            angle, k,
                        ) in angle_parameters[template_index][force_index]:

                        combined_force.addAngle(
                            particle_index_1 + particle_offset,
                            particle_index_2 + particle_offset,
                            particle_index_3 + particle_offset,
                            angle,
                            energy_scale * k,
                        )

            yield combined_force

class PeriodicTorsionForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.PeriodicTorsionForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve torsion parameters for all forces.
        torsion_parameters = [
            [
                [force.getTorsionParameters(torsion_index) for torsion_index in range(force.getNumTorsions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on whether or not they use periodic
        # boundary conditions.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault(force.usesPeriodicBoundaryConditions(), {}).setdefault(template_index, []).append(force_index)

        for uses_pbc, class_forces in force_classes.items():
            combined_force = openmm.PeriodicTorsionForce()
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for (
                            particle_index_1, particle_index_2, particle_index_3, particle_index_4,
                            periodicity, phase, k,
                        ) in torsion_parameters[template_index][force_index]:

                        combined_force.addTorsion(
                            particle_index_1 + particle_offset,
                            particle_index_2 + particle_offset,
                            particle_index_3 + particle_offset,
                            particle_index_4 + particle_offset,
                            periodicity,
                            phase,
                            energy_scale * k,
                        )

            yield combined_force

class RBTorsionForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.RBTorsionForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve torsion parameters for all forces.
        torsion_parameters = [
            [
                [force.getTorsionParameters(torsion_index) for torsion_index in range(force.getNumTorsions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on whether or not they use periodic
        # boundary conditions.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault(force.usesPeriodicBoundaryConditions(), {}).setdefault(template_index, []).append(force_index)

        for uses_pbc, class_forces in force_classes.items():
            combined_force = openmm.RBTorsionForce()
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for (
                            particle_index_1, particle_index_2, particle_index_3, particle_index_4,
                            c_0, c_1, c_2, c_3, c_4, c_5,
                        ) in torsion_parameters[template_index][force_index]:

                        combined_force.addTorsion(
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

            yield combined_force

class CMAPTorsionForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CMAPTorsionForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve CMAP map parameters for all forces.
        map_parameters = [
            [
                [force.getMapParameters(map_index) for map_index in range(force.getNumMaps())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve torsion parameters for all forces.
        torsion_parameters = [
            [
                [force.getTorsionParameters(torsion_index) for torsion_index in range(force.getNumTorsions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on whether or not they use periodic
        # boundary conditions.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault(force.usesPeriodicBoundaryConditions(), {}).setdefault(template_index, []).append(force_index)

        for uses_pbc, class_forces in force_classes.items():
            combined_force = openmm.CMAPTorsionForce()
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            # Keep track of the indices in the combined force of all of the maps
            # added so far, first directly by the (scaled) map parameters
            # themselves, and then by the indices and scale needed to look up
            # the map parameters in the map parameter table and scale them.
            map_parameters_table = {}
            map_indices_table = {}

            def get_map_index_by_indices(template_index, force_index, map_index, energy_scale):
                # Retrieves the index of a map in the combined force given
                # indices needed to look up a map in one of the template forces
                # and a factor by which its energy parameters should be scaled.

                map_indices_key = (template_index, force_index, map_index, energy_scale)
                map_indices_value = map_indices_table.get(map_indices_key, None)

                if map_indices_value is None:
                    # If a map has not already been found to be available in the
                    # combined force for these indices and scale, create one or
                    # look up a suitable existing one, store its index, and
                    # return this index.
                    map_indices_table[map_indices_key] = map_indices_value = get_map_index_by_parameters(*map_indices_key)

                return map_indices_value

            def get_map_index_by_parameters(template_index, force_index, map_index, energy_scale):
                # Retrieves the index of a map in the combined force given
                # indices needed to look up a map in one of the template forces
                # and a factor by which its energy parameters should be scaled.
                # If this function is called, no index exists for the given
                # indices and scale in the table of map indices by indices and
                # scale, but an index may exist in the table by parameters.

                size, energy = map_parameters[template_index][force_index][map_index]
                map_parameters_key = (size, tuple(energy_scale * energy_ij for energy_ij in energy.value_in_unit_system(openmm.unit.md_unit_system)))
                map_parameters_value = map_parameters_table.get(map_parameters_key, None)

                if map_parameters_value is None:
                    # If a map has not already been created in the combined
                    # force with these parameters, create one, store its index,
                    # and return this index.
                    map_parameters_table[map_parameters_key] = map_parameters_value = combined_force.getNumMaps()
                    combined_force.addMap(*map_parameters_key)

                return map_parameters_value

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for (
                            map_index,
                            particle_index_a_1, particle_index_a_2, particle_index_a_3, particle_index_a_4,
                            particle_index_b_1, particle_index_b_2, particle_index_b_3, particle_index_b_4,
                        ) in torsion_parameters[template_index][force_index]:

                        combined_force.addTorsion(
                            get_map_index_by_indices(template_index, force_index, map_index, energy_scale),
                            particle_index_a_1 + particle_offset,
                            particle_index_a_2 + particle_offset,
                            particle_index_a_3 + particle_offset,
                            particle_index_a_4 + particle_offset,
                            particle_index_b_1 + particle_offset,
                            particle_index_b_2 + particle_offset,
                            particle_index_b_3 + particle_offset,
                            particle_index_b_4 + particle_offset,
                        )

            yield combined_force

class CustomExternalForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomExternalForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve global parameter names for all forces.
        global_parameter_names = [
            [
                [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve global parameter values for all forces.
        global_parameter_values = [
            [
                [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve term parameter names for all forces.
        term_parameter_names = [
            [
                [force.getPerParticleParameterName(term_parameter_index) for term_parameter_index in range(force.getNumPerParticleParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve term parameters for all forces.
        term_parameters = [
            [
                [force.getParticleParameters(term_index) for term_index in range(force.getNumParticles())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on their energy functions, global
        # parameter names, global parameter values, and term parameter names.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault((
                    force.getEnergyFunction(),
                    tuple(global_parameter_names[template_index][force_index]),
                    tuple(global_parameter_values[template_index][force_index]),
                    tuple(term_parameter_names[template_index][force_index]),
                ), {}).setdefault(template_index, []).append(force_index)

        for (
                energy_function,
                class_global_parameter_names,
                class_global_parameter_values,
                class_term_parameter_names,
            ), class_forces in force_classes.items():

            scale_name = _get_scale_name(energy_function,
                class_global_parameter_names
                    + class_term_parameter_names
            )
            combined_force = openmm.CustomExternalForce(_scale_function(energy_function, scale_name))

            for global_parameter_name, global_parameter_value in zip(class_global_parameter_names, class_global_parameter_values):
                combined_force.addGlobalParameter(global_parameter_name, global_parameter_value)

            combined_force.addPerParticleParameter(scale_name)
            for term_parameter_name in class_term_parameter_names:
                combined_force.addPerParticleParameter(term_parameter_name)

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for particle_index, term_parameter_values in term_parameters[template_index][force_index]:
                        combined_force.addParticle(particle_index + particle_offset, [energy_scale, *term_parameter_values])

            yield combined_force

class CustomBondForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomBondForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve global parameter names for all forces.
        global_parameter_names = [
            [
                [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve global parameter values for all forces.
        global_parameter_values = [
            [
                [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve bond parameter names for all forces.
        bond_parameter_names = [
            [
                [force.getPerBondParameterName(bond_parameter_index) for bond_parameter_index in range(force.getNumPerBondParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve bond parameters for all forces.
        bond_parameters = [
            [
                [force.getBondParameters(bond_index) for bond_index in range(force.getNumBonds())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve derivative names for all forces.
        derivative_names = [
            [
                [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(force.getNumEnergyParameterDerivatives())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on their energy functions, whether or
        # not they use periodic boundary conditions, global parameter names,
        # global parameter values, bond parameter names, and derivative names.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault((
                    force.getEnergyFunction(),
                    force.usesPeriodicBoundaryConditions(),
                    tuple(global_parameter_names[template_index][force_index]),
                    tuple(global_parameter_values[template_index][force_index]),
                    tuple(bond_parameter_names[template_index][force_index]),
                    tuple(derivative_names[template_index][force_index]),
                ), {}).setdefault(template_index, []).append(force_index)

        for (
                energy_function,
                uses_pbc,
                class_global_parameter_names,
                class_global_parameter_values,
                class_bond_parameter_names,
                class_derivative_names,
            ), class_forces in force_classes.items():

            scale_name = _get_scale_name(energy_function,
                class_global_parameter_names
                    + class_bond_parameter_names
                    + class_derivative_names
            )
            combined_force = openmm.CustomBondForce(_scale_function(energy_function, scale_name))
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for global_parameter_name, global_parameter_value in zip(class_global_parameter_names, class_global_parameter_values):
                combined_force.addGlobalParameter(global_parameter_name, global_parameter_value)

            combined_force.addPerBondParameter(scale_name)
            for bond_parameter_name in class_bond_parameter_names:
                combined_force.addPerBondParameter(bond_parameter_name)

            for derivative_name in class_derivative_names:
                combined_force.addEnergyParameterDerivative(derivative_name)

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for (
                            particle_index_1, particle_index_2,
                            bond_parameter_values,
                        ) in bond_parameters[template_index][force_index]:

                        combined_force.addBond(
                            particle_index_1 + particle_offset,
                            particle_index_2 + particle_offset,
                            [energy_scale, *bond_parameter_values],
                        )

            yield combined_force

class CustomAngleForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomAngleForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve global parameter names for all forces.
        global_parameter_names = [
            [
                [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve global parameter values for all forces.
        global_parameter_values = [
            [
                [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve angle parameter names for all forces.
        angle_parameter_names = [
            [
                [force.getPerAngleParameterName(angle_parameter_index) for angle_parameter_index in range(force.getNumPerAngleParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve angle parameters for all forces.
        angle_parameters = [
            [
                [force.getAngleParameters(angle_index) for angle_index in range(force.getNumAngles())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve derivative names for all forces.
        derivative_names = [
            [
                [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(force.getNumEnergyParameterDerivatives())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on their energy functions, whether or
        # not they use periodic boundary conditions, global parameter names,
        # global parameter values, angle parameter names, and derivative names.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault((
                    force.getEnergyFunction(),
                    force.usesPeriodicBoundaryConditions(),
                    tuple(global_parameter_names[template_index][force_index]),
                    tuple(global_parameter_values[template_index][force_index]),
                    tuple(angle_parameter_names[template_index][force_index]),
                    tuple(derivative_names[template_index][force_index]),
                ), {}).setdefault(template_index, []).append(force_index)

        for (
                energy_function,
                uses_pbc,
                class_global_parameter_names,
                class_global_parameter_values,
                class_angle_parameter_names,
                class_derivative_names,
            ), class_forces in force_classes.items():

            scale_name = _get_scale_name(energy_function,
                class_global_parameter_names
                    + class_angle_parameter_names
                    + class_derivative_names
            )
            combined_force = openmm.CustomAngleForce(_scale_function(energy_function, scale_name))
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for global_parameter_name, global_parameter_value in zip(class_global_parameter_names, class_global_parameter_values):
                combined_force.addGlobalParameter(global_parameter_name, global_parameter_value)

            combined_force.addPerAngleParameter(scale_name)
            for angle_parameter_name in class_angle_parameter_names:
                combined_force.addPerAngleParameter(angle_parameter_name)

            for derivative_name in class_derivative_names:
                combined_force.addEnergyParameterDerivative(derivative_name)

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for (
                            particle_index_1, particle_index_2, particle_index_3,
                            angle_parameter_values,
                        ) in angle_parameters[template_index][force_index]:

                        combined_force.addAngle(
                            particle_index_1 + particle_offset,
                            particle_index_2 + particle_offset,
                            particle_index_3 + particle_offset,
                            [energy_scale, *angle_parameter_values],
                        )

            yield combined_force

class CustomTorsionForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomTorsionForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve global parameter names for all forces.
        global_parameter_names = [
            [
                [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve global parameter values for all forces.
        global_parameter_values = [
            [
                [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve torsion parameter names for all forces.
        torsion_parameter_names = [
            [
                [force.getPerTorsionParameterName(torsion_parameter_index) for torsion_parameter_index in range(force.getNumPerTorsionParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve torsion parameters for all forces.
        torsion_parameters = [
            [
                [force.getTorsionParameters(torsion_index) for torsion_index in range(force.getNumTorsions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve derivative names for all forces.
        derivative_names = [
            [
                [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(force.getNumEnergyParameterDerivatives())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on their energy functions, whether or
        # not they use periodic boundary conditions, global parameter names,
        # global parameter values, torsion parameter names, and derivative
        # names.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault((
                    force.getEnergyFunction(),
                    force.usesPeriodicBoundaryConditions(),
                    tuple(global_parameter_names[template_index][force_index]),
                    tuple(global_parameter_values[template_index][force_index]),
                    tuple(torsion_parameter_names[template_index][force_index]),
                    tuple(derivative_names[template_index][force_index]),
                ), {}).setdefault(template_index, []).append(force_index)

        for (
                energy_function,
                uses_pbc,
                class_global_parameter_names,
                class_global_parameter_values,
                class_torsion_parameter_names,
                class_derivative_names,
            ), class_forces in force_classes.items():

            scale_name = _get_scale_name(energy_function,
                class_global_parameter_names
                    + class_torsion_parameter_names
                    + class_derivative_names
            )
            combined_force = openmm.CustomTorsionForce(_scale_function(energy_function, scale_name))
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for global_parameter_name, global_parameter_value in zip(class_global_parameter_names, class_global_parameter_values):
                combined_force.addGlobalParameter(global_parameter_name, global_parameter_value)

            combined_force.addPerTorsionParameter(scale_name)
            for torsion_parameter_name in class_torsion_parameter_names:
                combined_force.addPerTorsionParameter(torsion_parameter_name)

            for derivative_name in class_derivative_names:
                combined_force.addEnergyParameterDerivative(derivative_name)

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for (
                            particle_index_1, particle_index_2, particle_index_3, particle_index_4,
                            torsion_parameter_values,
                        ) in torsion_parameters[template_index][force_index]:

                        combined_force.addTorsion(
                            particle_index_1 + particle_offset,
                            particle_index_2 + particle_offset,
                            particle_index_3 + particle_offset,
                            particle_index_4 + particle_offset,
                            [energy_scale, *torsion_parameter_values],
                        )

            yield combined_force

class CustomCompoundBondForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomCompoundBondForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve global parameter names for all forces.
        global_parameter_names = [
            [
                [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve global parameter values for all forces.
        global_parameter_values = [
            [
                [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve bond parameter names for all forces.
        bond_parameter_names = [
            [
                [force.getPerBondParameterName(bond_parameter_index) for bond_parameter_index in range(force.getNumPerBondParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve bond parameters for all forces.
        bond_parameters = [
            [
                [force.getBondParameters(bond_index) for bond_index in range(force.getNumBonds())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve derivative names for all forces.
        derivative_names = [
            [
                [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(force.getNumEnergyParameterDerivatives())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve tabulated function names for all forces.
        function_names = [
            [
                [force.getTabulatedFunctionName(function_index) for function_index in range(force.getNumTabulatedFunctions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve tabulated functions for all forces.
        functions = [
            [
                [force.getTabulatedFunction(function_index) for function_index in range(force.getNumTabulatedFunctions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on the number of particles per bond
        # that they use, their energy functions, whether or not they use
        # periodic boundary conditions, global parameter names, global parameter
        # values, bond parameter names, derivative names, tabulated function
        # names, and tabulated functions.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault((
                    force.getNumParticlesPerBond(),
                    force.getEnergyFunction(),
                    force.usesPeriodicBoundaryConditions(),
                    tuple(global_parameter_names[template_index][force_index]),
                    tuple(global_parameter_values[template_index][force_index]),
                    tuple(bond_parameter_names[template_index][force_index]),
                    tuple(derivative_names[template_index][force_index]),
                    tuple(function_names[template_index][force_index]),
                    tuple(DefaultTabulatedFunctionProcessor._process(function) for function in functions[template_index][force_index]),
                ), {}).setdefault(template_index, []).append(force_index)

        for (
                particles_per_bond,
                energy_function,
                uses_pbc,
                class_global_parameter_names,
                class_global_parameter_values,
                class_bond_parameter_names,
                class_derivative_names,
                class_function_names,
                class_functions,
            ), class_forces in force_classes.items():

            scale_name = _get_scale_name(energy_function,
                class_global_parameter_names
                    + class_bond_parameter_names
                    + class_derivative_names
                    + class_function_names
            )
            combined_force = openmm.CustomCompoundBondForce(particles_per_bond, _scale_function(energy_function, scale_name))
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for global_parameter_name, global_parameter_value in zip(class_global_parameter_names, class_global_parameter_values):
                combined_force.addGlobalParameter(global_parameter_name, global_parameter_value)

            combined_force.addPerBondParameter(scale_name)
            for bond_parameter_name in class_bond_parameter_names:
                combined_force.addPerBondParameter(bond_parameter_name)

            for derivative_name in class_derivative_names:
                combined_force.addEnergyParameterDerivative(derivative_name)

            for function_name, function in zip(class_function_names, class_functions):
                combined_force.addTabulatedFunction(function_name, DefaultTabulatedFunctionProcessor._process(function))

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    for particle_indices, bond_parameter_values in bond_parameters[template_index][force_index]:
                        combined_force.addBond(
                            [particle_index + particle_offset for particle_index in particle_indices],
                            [energy_scale, *bond_parameter_values],
                        )

            yield combined_force

class CustomCentroidBondForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomCentroidBondForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve global parameter names for all forces.
        global_parameter_names = [
            [
                [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve global parameter values for all forces.
        global_parameter_values = [
            [
                [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve bond parameter names for all forces.
        bond_parameter_names = [
            [
                [force.getPerBondParameterName(bond_parameter_index) for bond_parameter_index in range(force.getNumPerBondParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve group parameters for all forces.
        group_parameters = [
            [
                [force.getGroupParameters(group_index) for group_index in range(force.getNumGroups())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve bond parameters for all forces.
        bond_parameters = [
            [
                [force.getBondParameters(bond_index) for bond_index in range(force.getNumBonds())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve derivative names for all forces.
        derivative_names = [
            [
                [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(force.getNumEnergyParameterDerivatives())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve tabulated function names for all forces.
        function_names = [
            [
                [force.getTabulatedFunctionName(function_index) for function_index in range(force.getNumTabulatedFunctions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve tabulated functions for all forces.
        functions = [
            [
                [force.getTabulatedFunction(function_index) for function_index in range(force.getNumTabulatedFunctions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on the number of groups per bond that
        # they use, their energy functions, whether or not they use periodic
        # boundary conditions, global parameter names, global parameter values,
        # bond parameter names, derivative names, tabulated function names, and
        # tabulated functions.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault((
                    force.getNumGroupsPerBond(),
                    force.getEnergyFunction(),
                    force.usesPeriodicBoundaryConditions(),
                    tuple(global_parameter_names[template_index][force_index]),
                    tuple(global_parameter_values[template_index][force_index]),
                    tuple(bond_parameter_names[template_index][force_index]),
                    tuple(derivative_names[template_index][force_index]),
                    tuple(function_names[template_index][force_index]),
                    tuple(DefaultTabulatedFunctionProcessor._process(function) for function in functions[template_index][force_index]),
                ), {}).setdefault(template_index, []).append(force_index)

        for (
                groups_per_bond,
                energy_function,
                uses_pbc,
                class_global_parameter_names,
                class_global_parameter_values,
                class_bond_parameter_names,
                class_derivative_names,
                class_function_names,
                class_functions,
            ), class_forces in force_classes.items():

            scale_name = _get_scale_name(energy_function,
                class_global_parameter_names
                    + class_bond_parameter_names
                    + class_derivative_names
                    + class_function_names
            )
            combined_force = openmm.CustomCentroidBondForce(groups_per_bond, _scale_function(energy_function, scale_name))
            combined_force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for global_parameter_name, global_parameter_value in zip(class_global_parameter_names, class_global_parameter_values):
                combined_force.addGlobalParameter(global_parameter_name, global_parameter_value)

            combined_force.addPerBondParameter(scale_name)
            for bond_parameter_name in class_bond_parameter_names:
                combined_force.addPerBondParameter(bond_parameter_name)

            for derivative_name in class_derivative_names:
                combined_force.addEnergyParameterDerivative(derivative_name)

            for function_name, function in zip(class_function_names, class_functions):
                combined_force.addTabulatedFunction(function_name, DefaultTabulatedFunctionProcessor._process(function))

            for template_index, particle_offset, energy_scale in zip(template_indices, particle_offsets, energy_scales):
                for force_index in class_forces.get(template_index, []):
                    group_offset = combined_force.getNumGroups()

                    for particle_indices, weights in group_parameters[template_index][force_index]:
                        combined_force.addGroup(
                            [particle_index + particle_offset for particle_index in particle_indices],
                            weights,
                        )

                    for group_indices, bond_parameter_values in bond_parameters[template_index][force_index]:
                        combined_force.addBond(
                            [group_index + group_offset for group_index in group_indices],
                            [energy_scale, *bond_parameter_values],
                        )

            yield combined_force

class CustomNonbondedForceHandler(ForceHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.CustomNonbondedForce,)

    def handle(self, force_table, template_indices, particle_offsets, energy_scales):
        # Retrieve computed value parameters for all forces.
        compute_parameters = [
            [
                [tuple(force.getComputedValueParameters(compute_index)) for compute_index in range(force.getNumComputedValues())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve global parameter names for all forces.
        global_parameter_names = [
            [
                [force.getGlobalParameterName(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve global parameter values for all forces.
        global_parameter_values = [
            [
                [force.getGlobalParameterDefaultValue(global_parameter_index) for global_parameter_index in range(force.getNumGlobalParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve particle parameter names for all forces.
        particle_parameter_names = [
            [
                [force.getPerParticleParameterName(particle_parameter_index) for particle_parameter_index in range(force.getNumPerParticleParameters())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve particle parameters for all forces.
        particle_parameters = [
            [
                [force.getParticleParameters(particle_index) for particle_index in range(force.getNumParticles())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve interaction group parameters for all forces.
        group_parameters = [
            [
                # If interaction groups are present, retrieve them.  If no
                # interaction groups are present, add a single interaction group
                # containing all particles (this will not change the effective
                # behavior of the force).  In this way, the interaction group
                # list for a force can always be consulted to determine the
                # particle pairs that should be interacting.
                [force.getInteractionGroupParameters(group_index) for group_index in range(force.getNumInteractionGroups())]
                if force.getNumInteractionGroups() else [[tuple(range(force.getNumParticles()))] * 2]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve exclusion parameters for all forces.
        exclusion_parameters = [
            [
                [force.getExclusionParticles(exclusion_index) for exclusion_index in range(force.getNumExclusions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve derivative names for all forces.
        derivative_names = [
            [
                [force.getEnergyParameterDerivativeName(derivative_index) for derivative_index in range(force.getNumEnergyParameterDerivatives())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve tabulated function names for all forces.
        function_names = [
            [
                [force.getTabulatedFunctionName(function_index) for function_index in range(force.getNumTabulatedFunctions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Retrieve tabulated functions for all forces.
        functions = [
            [
                [force.getTabulatedFunction(function_index) for function_index in range(force.getNumTabulatedFunctions())]
                for force in template_forces
            ]
            for template_forces in force_table
        ]

        # Split forces into classes based on their energy functions, nonbonded
        # methods, whether or not they use switching, whether or not they use
        # long-range correction, cutoff distances, switching distances, computed
        # value parameters, global parameter names, global parameter values,
        # particle parameter names, derivative names, tabulated function names,
        # and tabulated functions.
        force_classes = {}
        for template_index, template_forces in enumerate(force_table):
            for force_index, force in enumerate(template_forces):
                force_classes.setdefault((
                    force.getEnergyFunction(),
                    force.getNonbondedMethod(),
                    force.getUseSwitchingFunction(),
                    force.getUseLongRangeCorrection(),
                    force.getCutoffDistance().value_in_unit_system(openmm.unit.md_unit_system),
                    force.getSwitchingDistance().value_in_unit_system(openmm.unit.md_unit_system),
                    tuple(compute_parameters[template_index][force_index]),
                    tuple(global_parameter_names[template_index][force_index]),
                    tuple(global_parameter_values[template_index][force_index]),
                    tuple(particle_parameter_names[template_index][force_index]),
                    tuple(derivative_names[template_index][force_index]),
                    tuple(function_names[template_index][force_index]),
                    tuple(DefaultTabulatedFunctionProcessor._process(function) for function in functions[template_index][force_index]),
                ), {}).setdefault(template_index, []).append(force_index)
        
        for (
                energy_function,
                nonbonded_method,
                uses_switch,
                uses_long_range,
                cut_distance,
                switch_distance,
                class_compute_parameters,
                class_global_parameter_names,
                class_global_parameter_values,
                class_particle_parameter_names,
                class_derivative_names,
                class_function_names,
                class_functions,
            ), class_forces in force_classes.items():

            scale_name = _get_scale_name(energy_function,
                tuple(compute_name for compute_name, compute in class_compute_parameters)
                    + class_global_parameter_names
                    + class_particle_parameter_names
                    + class_derivative_names
                    + class_function_names
            )

            # If multiple forces of this class are present in one or more
            # templates, multiple forces will need to be added to the combined
            # system to handle differing particle parameter values and
            # exclusions in general.
            for class_force_index in range(max(map(len, class_forces.values()), default=0)):
                combined_force = openmm.CustomNonbondedForce(_scale_function(energy_function, f"{scale_name}1"))
                combined_force.setNonbondedMethod(nonbonded_method)
                combined_force.setUseSwitchingFunction(uses_switch)
                combined_force.setUseLongRangeCorrection(uses_long_range)
                combined_force.setCutoffDistance(cut_distance)
                combined_force.setSwitchingDistance(switch_distance)

                for compute_name, compute in class_compute_parameters:
                    combined_force.addComputedValue(compute_name, compute)

                for global_parameter_name, global_parameter_value in zip(class_global_parameter_names, class_global_parameter_values):
                    combined_force.addGlobalParameter(global_parameter_name, global_parameter_value)

                combined_force.addPerParticleParameter(scale_name)
                for particle_parameter_name in class_particle_parameter_names:
                    combined_force.addPerParticleParameter(particle_parameter_name)

                for derivative_name in class_derivative_names:
                    combined_force.addEnergyParameterDerivative(derivative_name)

                for function_name, function in zip(class_function_names, class_functions):
                    combined_force.addTabulatedFunction(function_name, DefaultTabulatedFunctionProcessor._process(function))

                for template_index, particle_offset, next_particle_offset, energy_scale in zip(template_indices, particle_offsets, particle_offsets[1:], energy_scales):
                    if class_force_index < len(class_forces.get(template_index, [])):
                        force_index = class_forces[template_index][class_force_index]

                        for particle_parameter_values in particle_parameters[template_index][force_index]:
                            combined_force.addParticle([energy_scale, *particle_parameter_values])

                        for particle_indices_1, particle_indices_2 in group_parameters[template_index][force_index]:
                            combined_force.addInteractionGroup(
                                [particle_index_1 + particle_offset for particle_index_1 in particle_indices_1],
                                [particle_index_2 + particle_offset for particle_index_2 in particle_indices_2],
                            )

                        for particle_index_1, particle_index_2 in exclusion_parameters[template_index][force_index]:
                            combined_force.addExclusion(
                                particle_index_1 + particle_offset,
                                particle_index_2 + particle_offset,
                            )

                    else:
                        # Add dummy particles to the combined force so that the
                        # number of particles matches that of the combined
                        # system.  These particles will not be present in any
                        # interaction group.
                        for particle_index in range(next_particle_offset - particle_offset):
                            combined_force.addParticle([energy_scale] + [0] * len(class_particle_parameter_names))
                
                yield combined_force

class TabulatedFunctionProcessor(Processor):
    """
    Represents a processor for dispatching the handling of
    :py:class:`openmm.openmm.TabulatedFunction` objects to registered handler
    objects.

    .. py:method:: register_handler(handler)

        Registers a tabulated function handler.  All tabulated function types
        that the handler reports as handleable will be mapped to the handler,
        overriding the mappings of any previously registered handlers also
        reporting one or more of the same tabulated function types.

        :type handler: multiopenmm.stacking.TabulatedFunctionHandler
        :param handler: The handler to register.
    """

    __slots__ = ()

    def _check_handler(self, handler):
        if not isinstance(handler, TabulatedFunctionHandler):
            raise TypeError("handler must be a TabulatedFunctionHandler")

    def _process(self, function):
        handler_table = self._handler_table
        
        # If a tuple is received, a tabulated function object should be
        # reconstructed from the given type and data.  If a tabulated function
        # object is received, it should be deconstructed into a tuple containing
        # its type and data generated by a handler.
        reconstruct = isinstance(function, tuple)
        if reconstruct:
            function_type, function_data = function
        else:
            function_type = type(function)
            function_data = function

        # When processing a tabulated function, search through its method
        # resolution order list such that the handler table will be searched
        # first for the type of the tabulated function itself, followed by base
        # classes.
        for base_type in function_type.mro():
            if base_type in handler_table:
                if reconstruct:
                    return handler_table[base_type].reconstruct(function_type, function_data)
                else:
                    return (function_type, handler_table[base_type].deconstruct(function))
        else:
            raise TypeError(f"{function_type.__name__} is unsupported")

class TabulatedFunctionHandler(abc.ABC):
    """
    An abstract class that can be inherited from to define a custom handler for
    one or more :py:class:`openmm.openmm.TabulatedFunction` subclasses.

    Once such a handler has been defined, an instance of it must be registered,
    *e.g.*:

    .. code-block::

        class MyCustomTabulatedFunctionHandler(multiopenmm.stacking.TabulatedFunctionHandler):
            @property
            def handled_types(self):
                return (MyCustomOpenMMTabulatedFunction,)

            def deconstruct(self, function):
                ...
                return ...

            def reconstruct(self, function_type, function_data):
                ...
                return MyCustomOpenMMTabulatedFunction(...)

        multiopenmm.stacking.DefaultTabulatedFunctionProcessor.register_handler(MyCustomTabulatedFunctionHandler())

    Notes
    -----
    It is trivial to implement :py:meth:`deconstruct` and :py:meth:`reconstruct`
    using :py:meth:`openmm.openmm.XmlSerializer.serialize` and
    :py:meth:`openmm.openmm.XmlSerializer.deserialize`, respectively.  In fact,
    this is the default behavior for tabulated function types without a
    registered handler.  This mechanism, however, allows for the registration of
    alternative handlers that can be much faster than
    :py:class:`openmm.openmm.XmlSerializer`.  This can be significant for
    combined systems having many instances, each containing forces with
    tabulated functions having large tables of values.
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def handled_types(self):
        """
        tuple(type): The types of objects that can be handled by this handler.
        Each type should be a subclass of
        :py:class:`openmm.openmm.TabulatedFunction`.

        Notes
        -----
        This property must be implemented in derived classes.  A handler only
        handling objects of a single type should return a tuple of length 1.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def deconstruct(self, function):
        """
        Losslessly creates an immutable representation of a tabulated function
        that can be used as a dictionary key.

        Parameters
        ----------
        function : openmm.openmm.TabulatedFunction
            The tabulated function to deconstruct.

        Returns
        -------
        object
            A hashable representation of the tabulated function containing
            sufficient information to create an exact copy of the tabulated
            function.

        Notes
        -----
        This method must be implemented in derived classes.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def reconstruct(self, function_type, function_data):
        """
        Recreates a tabulated function from an immutable representation
        previously created by :py:meth:`deconstruct`.

        Parameters
        ----------
        function_type : type
            The type of the tabulated function from which the deconstruction was
            created.
        function_data : object
            A representation of a tabulated function created by
            :py:meth:`deconstruct`.

        Returns
        -------
        openmm.openmm.TabulatedFunction
            An exact copy of the tabulated function originally provided to
            :py:meth:`deconstruct`.

        Notes
        -----
        This method must be implemented in derived classes.
        """

        raise NotImplementedError

class FallbackTabulatedFunctionHandler(TabulatedFunctionHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.TabulatedFunction,)

    def deconstruct(self, function):
        return openmm.XmlSerializer.serialize(function)

    def reconstruct(self, function_type, function_data):
        return openmm.XmlSerializer.deserialize(function_data)

class DiscreteFunctionHandler(TabulatedFunctionHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.Discrete1DFunction, openmm.Discrete2DFunction, openmm.Discrete3DFunction)

    def deconstruct(self, function):
        result = tuple(function.getFunctionParameters())

        # Discrete1DFunction has only a single parameter and returns it directly
        # instead of wrapping it in a list of length 1, but Discrete2DFunction
        # and Discrete3DFunction have multiple parameters and return them in a
        # list, so normalize the result as appropriate to allow tuple unpacking
        # during function reconstruction later.
        return (result,) if isinstance(function, openmm.Discrete1DFunction) else result

    def reconstruct(self, function_type, function_data):
        return function_type(*function_data)

class ContinuousFunctionHandler(TabulatedFunctionHandler):
    __slots__ = ()

    @property
    def handled_types(self):
        return (openmm.Continuous1DFunction, openmm.Continuous2DFunction, openmm.Continuous3DFunction)

    def deconstruct(self, function):
        return (tuple(function.getFunctionParameters()), function.getPeriodic())

    def reconstruct(self, function_type, function_data):
        function_parameters, periodic = function_data
        return function_type(*function_parameters, periodic=periodic)

def _get_scale_name(function, parameter_names, prefix="scale"):
    # Retrieves an appropriate name for a scale parameter (such that the name
    # does not appear anywhere as a substring of the given function string and
    # that it is not equal to the names of any of the given parameters).

    parameter_names = set(parameter_names)
    for length in itertools.count():
        for suffix in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=length):
            name = prefix + "".join(map(str, suffix))
            if name not in parameter_names and name not in function:
                return name

def _scale_function(function, scale_name):
    # Returns the given function string scaled by a scale parameter with the
    # given name (handling function strings with additional definitions
    # delimited by semicolons after the primary expression).

    expression, *definitions = function.split(";")
    return ";".join([f"{scale_name}*({expression})", *definitions])

#: multiopenmm.stacking.VirtualSiteProcessor: The virtual site processor used by
#: :py:func:`multiopenmm.stacking.stack` to process all virtual sites in OpenMM
#: systems.
DefaultVirtualSiteProcessor = VirtualSiteProcessor()

DefaultVirtualSiteProcessor.register_handler(TwoParticleAverageSiteHandler())
DefaultVirtualSiteProcessor.register_handler(ThreeParticleAverageSiteHandler())
DefaultVirtualSiteProcessor.register_handler(OutOfPlaneSiteHandler())
DefaultVirtualSiteProcessor.register_handler(LocalCoordinatesSiteHandler())

#: multiopenmm.stacking.ForceProcessor: The force processor used by
#: :py:func:`multiopenmm.stacking.stack` to process all forces in OpenMM
#: systems.
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

#: multiopenmm.stacking.TabulatedFunctionProcessor: The tabulated function
#: processor used by :py:func:`multiopenmm.stacking.stack` to process all
#: tabulated functions in forces in OpenMM systems.
DefaultTabulatedFunctionProcessor = TabulatedFunctionProcessor()

DefaultTabulatedFunctionProcessor.register_handler(FallbackTabulatedFunctionHandler())
DefaultTabulatedFunctionProcessor.register_handler(DiscreteFunctionHandler())
DefaultTabulatedFunctionProcessor.register_handler(ContinuousFunctionHandler())
