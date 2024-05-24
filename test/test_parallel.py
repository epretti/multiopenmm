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

import helpers_test
import io
import multiopenmm
import numpy
import openmm
import pytest
import scipy
import tempfile

HELP_TEMPLATE_SIZES = (
    (),
    (0,),
    (1,),
    (1000,),
    (0, 0),
    (0, 1),
    (0, 1000),
    (1, 1),
    (1, 1000),
    (1000, 1000),
    (1000, 2000),
    (1000, 2000, 3000),
    (1000, 2000, 1000, 1000, 3000, 0, 4000, 0, 0, 5000, 6000),
)

HELP_RNG_SEED_FACTORIES = (
    lambda: 1,
    lambda: (2, 3, 4),
    lambda: numpy.random.SeedSequence(5),
    lambda: numpy.random.SFC64(6),
    lambda: numpy.random.default_rng(7),
)

HELP_REORDER_NULL = (
    (0, 0),
    (7, 7),
    (slice(1, 8, 3), slice(1, 8, 3)),
    (slice(5, 5), slice(4, 4)),
    ([], []),
    (numpy.zeros(0, dtype=int), numpy.zeros(0, dtype=int)),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([1, 3, 5, 4, 2], [1, 3, 5, 4, 2]),
    ([False, True, True, False, True, False, True, False], [False, True, True, False, True, False, True, False]),
    ([False] * 8, slice(4, 4)),
    ([2, 3, 5], [False, False, True, True, False, True, False, False]),
)

HELP_REORDER = (
    ([3, 4, 5], [4, 5, 3], [4, 5, 3], [3, 4, 5], ((0, 1), (102, 202), (1, 102), (202, 602))),
    (slice(2, 6), slice(5, 1, -1), [5, 4, 3, 2], [2, 3, 4, 5], ((102, 202), (2, 102), (1, 2), (0, 1), (202, 602))),
    ([False, False, False, False, True, True, True, True], [7, 4, 6, 5], [7, 4, 6, 5], [4, 5, 6, 7], ((0, 2), (102, 202), (402, 602), (202, 402), (2, 102))),
    (2, [3, 4, 5], [3, 4, 5], [2, 2, 2], ((0, 1), (0, 1), (0, 1), (0, 1), (202, 602))),
    ([3, 4, 5], [2, 2, 2], [2, 2, 2], [3, 4, 5], ((102, 202), (1, 602))),
)

HELP_INDICES_SPEC = (
    (0, None, []),
    (0, [], []),
    (4, 0, [0]),
    (4, 1, [1]),
    (4, -1, [3]),
    (1, -1, [0]),
    (1, None, [0]),
    (3, None, [0, 1, 2]),
    (4, slice(None), [0, 1, 2, 3]),
    (10, slice(1, 7, 2), [1, 3, 5]),
    (10, slice(9, 4, -1), [9, 8, 7, 6, 5]),
    (10, [2], [2]),
    (10, [2, 6, 4], [2, 6, 4]),
    (5, [True, False, True, False, True], [0, 2, 4]),
    (5, [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]),
    (5, [2, 2, -3, -3, 2], [2, 2, 2, 2, 2]),
    (5, [1, 2, 2, 2, 3], [1, 2, 2, 2, 3]),
)

HELP_EPSILON = numpy.sqrt(numpy.finfo(numpy.single).eps * numpy.finfo(numpy.double).eps)

HELP_K_B = openmm.unit.BOLTZMANN_CONSTANT_kB.value_in_unit_system(openmm.unit.md_unit_system) * \
    openmm.unit.AVOGADRO_CONSTANT_NA.value_in_unit_system(openmm.unit.md_unit_system) 

HELP_P_LIMIT = 1e-3

def help_get_rng_bytes(simulation):
    return simulation._Simulation__rng.bytes(256)

@pytest.mark.parametrize("precision", multiopenmm.Precision)
def test_precision_types(precision):
    assert numpy.issubdtype(precision.type, numpy.floating)

@pytest.mark.parametrize("template_sizes", HELP_TEMPLATE_SIZES)
def test_simulation_init_iterable(template_sizes):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    assert simulation.template_count == len(template_sizes)
    helpers_test.help_check_equal(simulation.get_template_sizes(), numpy.array(template_sizes))

@pytest.mark.parametrize("template_sizes", HELP_TEMPLATE_SIZES)
def test_simulation_init_tuple(template_sizes):
    simulation = multiopenmm.Simulation(tuple(helpers_test.help_make_templates(template_sizes)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    assert simulation.template_count == len(template_sizes)
    helpers_test.help_check_equal(simulation.get_template_sizes(), numpy.array(template_sizes))

@pytest.mark.parametrize("template_sizes", HELP_TEMPLATE_SIZES)
def test_simulation_init_list(template_sizes):
    simulation = multiopenmm.Simulation(list(helpers_test.help_make_templates(template_sizes)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    assert simulation.template_count == len(template_sizes)
    helpers_test.help_check_equal(simulation.get_template_sizes(), numpy.array(template_sizes))

@pytest.mark.parametrize("templates", (None, object(), openmm.System(), (openmm.System(), None)))
def test_simulation_init_templates_type(templates):
    with pytest.raises(TypeError):
        multiopenmm.Simulation(templates, multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())

@pytest.mark.parametrize("manager", (None, object(), multiopenmm.CanonicalEnsemble()))
def test_simulation_init_manager_type(manager):
    with pytest.raises(TypeError):
        multiopenmm.Simulation((), manager, multiopenmm.CanonicalEnsemble())

@pytest.mark.parametrize("ensemble", (None, object(), multiopenmm.SynchronousManager()))
def test_simulation_init_ensemble_type(ensemble):
    with pytest.raises(TypeError):
        multiopenmm.Simulation((), multiopenmm.SynchronousManager(), ensemble)

@pytest.mark.parametrize("precision", (None, object(), numpy.single, numpy.double))
def test_simulation_init_precision_type(precision):
    with pytest.raises(TypeError):
        multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), precision=precision)

@pytest.mark.parametrize("seed_factory", HELP_RNG_SEED_FACTORIES)
def test_simulation_init_seed(seed_factory):
    default_simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    seeded_simulation_1 = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), seed=seed_factory())
    seeded_simulation_2 = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), seed=seed_factory())

    default_bytes = help_get_rng_bytes(default_simulation)
    seeded_bytes_1 = help_get_rng_bytes(seeded_simulation_1)
    seeded_bytes_2 = help_get_rng_bytes(seeded_simulation_2)

    assert default_bytes != seeded_bytes_1
    assert seeded_bytes_1 == seeded_bytes_2

@pytest.mark.parametrize("seed", (object(), 1.0, "seed"))
def test_simulation_init_seed_type(seed):
    with pytest.raises(TypeError):
        multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), seed=seed)

@pytest.mark.parametrize("precision", multiopenmm.Precision)
def test_simulation_precision(precision):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), precision=precision)
    assert simulation.precision == precision

@pytest.mark.parametrize("ensemble", (multiopenmm.CanonicalEnsemble(), multiopenmm.IsothermalIsobaricEnsemble()))
def test_simulation_property_types(ensemble):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), ensemble)
    assert simulation.property_types == ensemble.property_types

@pytest.mark.parametrize("instance_count", (None, object(), 1.0))
def test_simulation_instance_count_type(instance_count):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.instance_count = instance_count

@pytest.mark.parametrize("instance_count", (-2, -1))
def test_simulation_instance_count_value(instance_count):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(ValueError):
        simulation.instance_count = instance_count

@pytest.mark.parametrize("instance_count", (0, 1, 2, 3))
def test_simulation_instance_count_no_templates(instance_count):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    if instance_count:
        with pytest.raises(ValueError):
            simulation.instance_count = instance_count
    else:
        simulation.instance_count = instance_count
        assert simulation.instance_count == instance_count

@pytest.mark.parametrize("instance_count", (0, 1, 2, 3))
def test_simulation_instance_count_templates(instance_count):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = instance_count
    assert simulation.instance_count == instance_count

@pytest.mark.parametrize("instance_count", (0, 1, 2, 3))
@pytest.mark.parametrize("template_size", (0, 1, 100))
def test_simulation_instance_count_set(instance_count, template_size):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((template_size, 1000)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = instance_count

    assert simulation.get_template_indices().shape == (instance_count,)
    assert simulation.get_instance_sizes().shape == (instance_count,)
    assert simulation.get_stacks().shape == (instance_count,)
    assert simulation.get_vectors().shape == (instance_count, 3, 3)
    assert simulation.get_positions().shape == (template_size * instance_count, 3)
    assert simulation.get_velocities().shape == (template_size * instance_count, 3)

    for name in simulation.property_types:
        assert simulation.get_property_values(name).shape == (instance_count,)

@pytest.mark.parametrize("instance_count", (0, 1, 2, 9, 10, 11, 20))
@pytest.mark.parametrize("template_size", (0, 1, 2, 100, 200))
def test_simulation_instance_count_resize(instance_count, template_size):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((template_size, 50, 150)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 10
    simulation.set_template_indices((1, 2) * 5)

    reference_stacks = 2 + 3 * numpy.arange(10)
    reference_vectors = 5 + 7 * numpy.arange(90).reshape(10, 3, 3)
    reference_positions = [block[start:stop] for block in 11 + 13 * numpy.arange(3000).reshape(5, 200, 3) for start, stop in ((0, 50), (50, 200))]
    reference_velocities = [block[start:stop] for block in 17 + 19 * numpy.arange(3000).reshape(5, 200, 3) for start, stop in ((0, 50), (50, 200))]
    reference_step_lengths = 23 + 29 * numpy.arange(10)

    simulation.set_stacks(reference_stacks)
    simulation.set_vectors(reference_vectors)
    simulation.set_positions(numpy.concatenate(reference_positions))
    simulation.set_velocities(numpy.concatenate(reference_velocities))
    simulation.set_property_values("step_length", reference_step_lengths)

    for index in range(10):
        helpers_test.help_check_equal(simulation.get_template_indices(index), numpy.array(((1, 2)[index % 2],)))
        helpers_test.help_check_equal(simulation.get_instance_sizes(index), numpy.array(((50, 150)[index % 2],)))
        helpers_test.help_check_equal(simulation.get_stacks(index), numpy.array((reference_stacks[index],)))
        helpers_test.help_check_equal(simulation.get_vectors(index), numpy.array((reference_vectors[index],)))
        helpers_test.help_check_equal(simulation.get_positions(index), reference_positions[index])
        helpers_test.help_check_equal(simulation.get_velocities(index), reference_velocities[index])
        helpers_test.help_check_equal(simulation.get_property_values("step_length", index), numpy.array((reference_step_lengths[index],)))

    simulation.instance_count = instance_count

    for index in range(instance_count):
        helpers_test.help_check_equal(simulation.get_template_indices(index), numpy.array(((1, 2)[index % 2] if index < 10 else 0,)))
        helpers_test.help_check_equal(simulation.get_instance_sizes(index), numpy.array(((50, 150)[index % 2] if index < 10 else template_size,)))
        helpers_test.help_check_equal(simulation.get_stacks(index), numpy.array((reference_stacks[index] if index < 10 else 0,)))
        helpers_test.help_check_equal(simulation.get_vectors(index), numpy.array((reference_vectors[index] if index < 10 else multiopenmm.parallel.DEFAULT_VECTOR_LENGTH * numpy.eye(3),)))
        helpers_test.help_check_equal(simulation.get_positions(index), reference_positions[index] if index < 10 else numpy.zeros((template_size, 3)))
        helpers_test.help_check_equal(simulation.get_velocities(index), reference_velocities[index] if index < 10 else numpy.zeros((template_size, 3)))
        helpers_test.help_check_equal(simulation.get_property_values("step_length", index), numpy.array((reference_step_lengths[index] if index < 10 else simulation.property_defaults.get("step_length", 0),)))

    simulation.instance_count = 10

    for index in range(10):
        helpers_test.help_check_equal(simulation.get_template_indices(index), numpy.array(((1, 2)[index % 2] if index < instance_count else 0,)))
        helpers_test.help_check_equal(simulation.get_instance_sizes(index), numpy.array(((50, 150)[index % 2] if index < instance_count else template_size,)))
        helpers_test.help_check_equal(simulation.get_stacks(index), numpy.array((reference_stacks[index] if index < instance_count else 0,)))
        helpers_test.help_check_equal(simulation.get_vectors(index), numpy.array((reference_vectors[index] if index < instance_count else multiopenmm.parallel.DEFAULT_VECTOR_LENGTH * numpy.eye(3),)))
        helpers_test.help_check_equal(simulation.get_positions(index), reference_positions[index] if index < instance_count else numpy.zeros((template_size, 3)))
        helpers_test.help_check_equal(simulation.get_velocities(index), reference_velocities[index] if index < instance_count else numpy.zeros((template_size, 3)))
        helpers_test.help_check_equal(simulation.get_property_values("step_length", index), numpy.array((reference_step_lengths[index] if index < instance_count else simulation.property_defaults.get("step_length", 0),)))

@pytest.mark.parametrize("template_size", (0, 1, 2, 100))
def test_simulation_reorder_instances_single(template_size):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((template_size,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 1

    reference_stacks = numpy.array((2,))
    reference_vectors = 5 + 7 * numpy.arange(9).reshape(1, 3, 3)
    reference_positions = 11 + 13 * numpy.arange(3 * template_size).reshape(template_size, 3)
    reference_velocities = 17 + 19 * numpy.arange(3 * template_size).reshape(template_size, 3)
    reference_step_lengths = numpy.array((23,))

    simulation.set_stacks(reference_stacks)
    simulation.set_vectors(reference_vectors)
    simulation.set_positions(reference_positions)
    simulation.set_velocities(reference_velocities)
    simulation.set_property_values("step_length", reference_step_lengths)

    simulation.reorder_instances(0, 0)

    helpers_test.help_check_equal(simulation.get_template_indices(), numpy.array((0,)))
    helpers_test.help_check_equal(simulation.get_instance_sizes(), numpy.array((template_size,)))
    helpers_test.help_check_equal(simulation.get_stacks(), reference_stacks)
    helpers_test.help_check_equal(simulation.get_vectors(), reference_vectors)
    helpers_test.help_check_equal(simulation.get_positions(), reference_positions)
    helpers_test.help_check_equal(simulation.get_velocities(), reference_velocities)
    helpers_test.help_check_equal(simulation.get_property_values("step_length"), reference_step_lengths)

@pytest.mark.parametrize("indices", HELP_REORDER_NULL)
def test_simulation_reorder_instances_multiple_null(indices):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0, 1, 100, 200)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 8
    simulation.set_template_indices((0, 0, 1, 1, 2, 2, 3, 3))

    reference_stacks = 2 + 3 * numpy.arange(8)
    reference_vectors = 5 + 7 * numpy.arange(72).reshape(8, 3, 3)
    reference_positions = 11 + 13 * numpy.arange(1806).reshape(602, 3)
    reference_velocities = 17 + 19 * numpy.arange(1806).reshape(602, 3)
    reference_step_lengths = 23 + 29 * numpy.arange(8)

    simulation.set_stacks(reference_stacks)
    simulation.set_vectors(reference_vectors)
    simulation.set_positions(reference_positions)
    simulation.set_velocities(reference_velocities)
    simulation.set_property_values("step_length", reference_step_lengths)

    simulation.reorder_instances(*indices)

    helpers_test.help_check_equal(simulation.get_template_indices(), numpy.array((0, 0, 1, 1, 2, 2, 3, 3)))
    helpers_test.help_check_equal(simulation.get_instance_sizes(), numpy.array((0, 0, 1, 1, 100, 100, 200, 200)))
    helpers_test.help_check_equal(simulation.get_stacks(), reference_stacks)
    helpers_test.help_check_equal(simulation.get_vectors(), reference_vectors)
    helpers_test.help_check_equal(simulation.get_positions(), reference_positions)
    helpers_test.help_check_equal(simulation.get_velocities(), reference_velocities)
    helpers_test.help_check_equal(simulation.get_property_values("step_length"), reference_step_lengths)

@pytest.mark.parametrize("reorder_data", HELP_REORDER)
def test_simulation_reorder_instances_multiple(reorder_data):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0, 1, 100, 200)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 8

    reference_indices = numpy.array((0, 0, 1, 1, 2, 2, 3, 3))
    reference_sizes = numpy.array((0, 0, 1, 1, 100, 100, 200, 200))
    reference_stacks = 2 + 3 * numpy.arange(8)
    reference_vectors = 5 + 7 * numpy.arange(72).reshape(8, 3, 3)
    reference_positions = 11 + 13 * numpy.arange(1806).reshape(602, 3)
    reference_velocities = 17 + 19 * numpy.arange(1806).reshape(602, 3)
    reference_step_lengths = 23 + 29 * numpy.arange(8)

    simulation.set_template_indices(reference_indices)
    simulation.set_stacks(reference_stacks)
    simulation.set_vectors(reference_vectors)
    simulation.set_positions(reference_positions)
    simulation.set_velocities(reference_velocities)
    simulation.set_property_values("step_length", reference_step_lengths)

    def check():
        helpers_test.help_check_equal(simulation.get_template_indices(), reference_indices)
        helpers_test.help_check_equal(simulation.get_instance_sizes(), reference_sizes)
        helpers_test.help_check_equal(simulation.get_stacks(), reference_stacks)
        helpers_test.help_check_equal(simulation.get_vectors(), reference_vectors)
        helpers_test.help_check_equal(simulation.get_positions(), reference_positions)
        helpers_test.help_check_equal(simulation.get_velocities(), reference_velocities)
        helpers_test.help_check_equal(simulation.get_property_values("step_length"), reference_step_lengths)

    source_indices, destination_indices, instance_to, instance_from, particle_slices = reorder_data
    simulation.reorder_instances(source_indices, destination_indices)
    for array in (reference_indices, reference_sizes, reference_stacks, reference_vectors, reference_step_lengths):
        array[instance_to] = array[instance_from]
    reference_positions = numpy.concatenate([reference_positions[slice_from:slice_to] for slice_from, slice_to in particle_slices])
    reference_velocities = numpy.concatenate([reference_velocities[slice_from:slice_to] for slice_from, slice_to in particle_slices])

    check()

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_get_template_sizes(indices_spec):
    count, indices_in, indices_out = indices_spec
    template_sizes = 2 + 3 * numpy.arange(count)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    helpers_test.help_check_equal(simulation.get_template_sizes(indices_in), template_sizes[indices_out])

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_get_instance_sizes(indices_spec):
    count, indices_in, indices_out = indices_spec
    template_sizes = 2 + 3 * numpy.arange(count)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(numpy.arange(count))
    helpers_test.help_check_equal(simulation.get_instance_sizes(indices_in), template_sizes[indices_out])

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_get_template_indices(indices_spec):
    count, indices_in, indices_out = indices_spec
    template_indices = 2 + 3 * numpy.arange(count)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,) * (numpy.amax(template_indices, initial=0) + 1)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(template_indices)
    helpers_test.help_check_equal(simulation.get_template_indices(indices_in), template_indices[indices_out])

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_get_stacks(indices_spec):
    count, indices_in, indices_out = indices_spec
    stacks = 2 + 3 * numpy.arange(count)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_stacks(stacks)
    helpers_test.help_check_equal(simulation.get_stacks(indices_in), stacks[indices_out])

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_get_vectors(indices_spec):
    count, indices_in, indices_out = indices_spec
    vectors = 2 + 3 * numpy.arange(9 * count).reshape(count, 3, 3)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_vectors(vectors)
    helpers_test.help_check_equal(simulation.get_vectors(indices_in), vectors[indices_out])

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_get_positions(indices_spec):
    count, indices_in, indices_out = indices_spec
    template_sizes = 2 + 3 * numpy.arange(count)
    positions = 5 + 7 * numpy.arange(3 * numpy.sum(template_sizes)).reshape(-1, 3)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(numpy.arange(count))
    simulation.set_positions(positions)
    helpers_test.help_check_equal(simulation.get_positions(indices_in), numpy.concatenate([positions[numpy.sum(template_sizes[:index]):numpy.sum(template_sizes[:index + 1])] for index in indices_out]) if indices_out else numpy.zeros((0, 3)))

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_get_velocities(indices_spec):
    count, indices_in, indices_out = indices_spec
    template_sizes = 2 + 3 * numpy.arange(count)
    velocities = 5 + 7 * numpy.arange(3 * numpy.sum(template_sizes)).reshape(-1, 3)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(numpy.arange(count))
    simulation.set_velocities(velocities)
    helpers_test.help_check_equal(simulation.get_velocities(indices_in), numpy.concatenate([velocities[numpy.sum(template_sizes[:index]):numpy.sum(template_sizes[:index + 1])] for index in indices_out]) if indices_out else numpy.zeros((0, 3)))

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_get_property_values(indices_spec):
    count, indices_in, indices_out = indices_spec
    step_lengths = 2 + 3 * numpy.arange(count)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_property_values("step_length", step_lengths)
    helpers_test.help_check_equal(simulation.get_property_values("step_length", indices_in), step_lengths[indices_out])

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
@pytest.mark.parametrize("broadcast", (False, True))
def test_simulation_set_template_indices(indices_spec, broadcast):
    count, indices_in, indices_out = indices_spec
    template_indices = 2 + 3 * numpy.arange(count)
    template_indices_new = 2 + 3 * (count if broadcast else numpy.arange(count, 2 * count))
    template_sizes = 5 + 7 * numpy.arange(numpy.amax(template_indices_new, initial=0) + 1)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(template_indices)

    positions = 11 + 13 * numpy.arange(3 * numpy.sum(simulation.get_instance_sizes())).reshape(-1, 3)
    simulation.set_positions(positions)
    old_positions = [simulation.get_positions(index) for index in range(count)]

    template_indices_old = numpy.array(template_indices)
    template_indices[indices_out] = template_indices_new if broadcast else template_indices_new[indices_out]
    helpers_test.help_check_equal(simulation.get_template_indices(), template_indices_old)
    simulation.set_template_indices(template_indices_new if broadcast else template_indices_new[indices_out], indices_in)
    helpers_test.help_check_equal(simulation.get_template_indices(), template_indices)

    for index in range(count):
        old_size = template_sizes[template_indices_old[index]]
        new_size = template_sizes[template_indices[index]]
        positions_check = numpy.zeros((new_size, 3))
        positions_check[:min(old_size, new_size)] = old_positions[index][:min(old_size, new_size)]
        helpers_test.help_check_equal(simulation.get_positions(index), positions_check)

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
@pytest.mark.parametrize("broadcast", (False, True))
def test_simulation_set_stacks(indices_spec, broadcast):
    count, indices_in, indices_out = indices_spec
    stacks = 2 + 3 * numpy.arange(count)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_stacks(stacks)

    stacks_old = numpy.array(stacks)
    stacks_new = 2 + 3 * (count if broadcast else numpy.arange(count, 2 * count))
    stacks[indices_out] = stacks_new if broadcast else stacks_new[indices_out]
    helpers_test.help_check_equal(simulation.get_stacks(), stacks_old)
    simulation.set_stacks(stacks_new if broadcast else stacks_new[indices_out], indices_in)
    helpers_test.help_check_equal(simulation.get_stacks(), stacks)

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
@pytest.mark.parametrize("broadcast", (False, True))
def test_simulation_set_vectors(indices_spec, broadcast):
    count, indices_in, indices_out = indices_spec
    vectors = 2 + 3 * numpy.arange(9 * count).reshape(count, 3, 3)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_vectors(vectors)

    vectors_old = numpy.array(vectors)
    vectors_new = 2 + 3 * (numpy.arange(9 * count, 9 * (count + 1)).reshape(3, 3) if broadcast else numpy.arange(9 * count, 18 * count).reshape(count, 3, 3))
    vectors[indices_out] = vectors_new if broadcast else vectors_new[indices_out]
    helpers_test.help_check_equal(simulation.get_vectors(), vectors_old)
    simulation.set_vectors(vectors_new if broadcast else vectors_new[indices_out], indices_in)
    helpers_test.help_check_equal(simulation.get_vectors(), vectors)

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_set_positions(indices_spec):
    count, indices_in, indices_out = indices_spec
    template_sizes = 2 + 3 * numpy.arange(count)
    positions = 5 + 7 * numpy.arange(3 * numpy.sum(template_sizes)).reshape(-1, 3)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(numpy.arange(count))
    simulation.set_positions(positions)

    positions_old = numpy.array(positions)
    positions_new = 5 + 7 * numpy.arange(3 * numpy.sum(template_sizes), 6 * numpy.sum(template_sizes)).reshape(-1, 3)
    for index in indices_out:
        start = numpy.sum(template_sizes[:index])
        end = numpy.sum(template_sizes[:index + 1])
        positions[start:end] = positions_new[start:end]
    helpers_test.help_check_equal(simulation.get_positions(), positions_old)
    simulation.set_positions(numpy.concatenate([positions_new[numpy.sum(template_sizes[:index]):numpy.sum(template_sizes[:index + 1])] for index in indices_out]) if indices_out else numpy.zeros((0, 3)), indices_in)
    helpers_test.help_check_equal(simulation.get_positions(), positions)

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_set_velocities(indices_spec):
    count, indices_in, indices_out = indices_spec
    template_sizes = 2 + 3 * numpy.arange(count)
    velocities = 5 + 7 * numpy.arange(3 * numpy.sum(template_sizes)).reshape(-1, 3)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(numpy.arange(count))
    simulation.set_velocities(velocities)

    velocities_old = numpy.array(velocities)
    velocities_new = 5 + 7 * numpy.arange(3 * numpy.sum(template_sizes), 6 * numpy.sum(template_sizes)).reshape(-1, 3)
    for index in indices_out:
        start = numpy.sum(template_sizes[:index])
        end = numpy.sum(template_sizes[:index + 1])
        velocities[start:end] = velocities_new[start:end]
    helpers_test.help_check_equal(simulation.get_velocities(), velocities_old)
    simulation.set_velocities(numpy.concatenate([velocities_new[numpy.sum(template_sizes[:index]):numpy.sum(template_sizes[:index + 1])] for index in indices_out]) if indices_out else numpy.zeros((0, 3)), indices_in)
    helpers_test.help_check_equal(simulation.get_velocities(), velocities)

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
@pytest.mark.parametrize("broadcast", (False, True))
def test_simulation_set_property_values(indices_spec, broadcast):
    count, indices_in, indices_out = indices_spec
    step_lengths = 2 + 3 * numpy.arange(count)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_property_values("step_length", step_lengths)

    step_lengths_old = numpy.array(step_lengths)
    step_lengths_new = 2 + 3 * (count if broadcast else numpy.arange(count, 2 * count))
    step_lengths[indices_out] = step_lengths_new if broadcast else step_lengths_new[indices_out]
    helpers_test.help_check_equal(simulation.get_property_values("step_length"), step_lengths_old)
    simulation.set_property_values("step_length", step_lengths_new if broadcast else step_lengths_new[indices_out], indices_in)
    helpers_test.help_check_equal(simulation.get_property_values("step_length"), step_lengths)

@pytest.mark.parametrize("count", (0, 1, 2, 10))
def test_simulation_set_stacks_separate(count):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_stacks_separate()
    assert len(set(simulation.get_stacks())) == count

@pytest.mark.parametrize("count", (0, 1, 2, 10))
def test_simulation_set_stacks_stacked(count):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((0,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_stacks_stacked()
    assert len(set(simulation.get_stacks())) == (1 if count else 0)

@pytest.mark.parametrize("file", (None, object(), "file", io.StringIO()))
def test_simulation_dump_file_type(file):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.dump(file)

@pytest.mark.parametrize("vectors", (None, object(), 0, 1))
def test_simulation_dump_vectors_type(vectors):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.dump(io.BytesIO(), vectors=vectors)

@pytest.mark.parametrize("positions", (None, object(), 0, 1))
def test_simulation_dump_positions_type(positions):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.dump(io.BytesIO(), positions=positions)

@pytest.mark.parametrize("velocities", (None, object(), 0, 1))
def test_simulation_dump_velocities_type(velocities):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.dump(io.BytesIO(), velocities=velocities)

@pytest.mark.parametrize("precision", (object(), numpy.single, numpy.double))
def test_simulation_dump_precision_type(precision):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.dump(io.BytesIO(), precision=precision)

@pytest.mark.parametrize("file", (None, object(), "file", io.StringIO()))
def test_simulation_load_file_type(file):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.load(file)

@pytest.mark.parametrize("vectors", (object(), 0, 1))
def test_simulation_load_vectors_type(vectors):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.load(io.BytesIO(), vectors=vectors)

@pytest.mark.parametrize("positions", (object(), 0, 1))
def test_simulation_load_positions_type(positions):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.load(io.BytesIO(), positions=positions)

@pytest.mark.parametrize("velocities", (object(), 0, 1))
def test_simulation_load_velocities_type(velocities):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.load(io.BytesIO(), velocities=velocities)

@pytest.mark.parametrize("precision", (object(), numpy.single, numpy.double))
def test_simulation_load_precision_type(precision):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.load(io.BytesIO(), precision=precision)

@pytest.mark.parametrize("do_vectors", (False, True))
@pytest.mark.parametrize("do_positions", (False, True))
@pytest.mark.parametrize("do_velocities", (False, True))
@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
def test_simulation_dump_load(do_vectors, do_positions, do_velocities, indices_spec):
    count, indices_in, indices_out = indices_spec
    template_sizes = 200 + 300 * numpy.arange(count)
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates(template_sizes), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(numpy.arange(count))

    old_vectors = 5 + 7 * numpy.arange(9 * count).reshape(count, 3, 3)
    old_positions = 11 + 13 * numpy.arange(3 * numpy.sum(template_sizes)).reshape(-1, 3)
    old_velocities = 17 + 19 * numpy.arange(3 * numpy.sum(template_sizes)).reshape(-1, 3)

    new_vectors = 23 + 29 * numpy.arange(9 * count).reshape(count, 3, 3)
    new_positions = 31 + 37 * numpy.arange(3 * numpy.sum(template_sizes)).reshape(-1, 3)
    new_velocities = 41 + 43 * numpy.arange(3 * numpy.sum(template_sizes)).reshape(-1, 3)

    simulation.set_vectors(old_vectors)
    simulation.set_positions(old_positions)
    simulation.set_velocities(old_velocities)

    data = io.BytesIO()
    simulation.dump(data, do_vectors, do_positions, do_velocities, indices=indices_in)

    simulation.set_vectors(new_vectors)
    simulation.set_positions(new_positions)
    simulation.set_velocities(new_velocities)

    data.seek(0)
    simulation.load(data, indices=indices_in)

    for index in range(count):
        in_out = index in indices_out
        helpers_test.help_check_equal(simulation.get_vectors(index), numpy.array(((old_vectors if do_vectors and in_out else new_vectors)[index],)))
        helpers_test.help_check_equal(simulation.get_positions(index), (old_positions if do_positions and in_out else new_positions)[numpy.sum(template_sizes[:index]):numpy.sum(template_sizes[:index + 1])])
        helpers_test.help_check_equal(simulation.get_velocities(index), (old_velocities if do_velocities and in_out else new_velocities)[numpy.sum(template_sizes[:index]):numpy.sum(template_sizes[:index + 1])])

@pytest.mark.parametrize("indices_spec", HELP_INDICES_SPEC)
@pytest.mark.parametrize("shuffle_out", (False, True))
def test_simulation_dump_load_shuffle(indices_spec, shuffle_out):
    count, indices_in, indices_out = indices_spec
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((100,) * count), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = count
    simulation.set_template_indices(numpy.arange(count))

    old_vectors = 5 + 7 * numpy.arange(9 * count).reshape(count, 3, 3)
    old_positions = 11 + 13 * numpy.arange(300 * count).reshape(-1, 3)
    old_velocities = 17 + 19 * numpy.arange(300 * count).reshape(-1, 3)

    new_vectors = 23 + 29 * numpy.arange(9 * count).reshape(count, 3, 3)
    new_positions = 31 + 37 * numpy.arange(300 * count).reshape(-1, 3)
    new_velocities = 41 + 43 * numpy.arange(300 * count).reshape(-1, 3)

    simulation.set_vectors(old_vectors)
    simulation.set_positions(old_positions)
    simulation.set_velocities(old_velocities)

    data = io.BytesIO()
    simulation.dump(data, indices=numpy.arange(len(indices_out)) if shuffle_out else indices_in)

    simulation.set_vectors(new_vectors)
    simulation.set_positions(new_positions)
    simulation.set_velocities(new_velocities)

    data.seek(0)
    simulation.load(data, indices=indices_in if shuffle_out else numpy.arange(len(indices_out)))

    for index in range(count):
        in_out = index in indices_out
        use_old = in_out if shuffle_out else index < len(indices_out)
        lookup_index = (len(indices_out) - 1 - indices_out[::-1].index(index) if in_out else index) if shuffle_out else (indices_out[index] if index < len(indices_out) else index)
        helpers_test.help_check_equal(simulation.get_vectors(index), numpy.array(((old_vectors if use_old else new_vectors)[lookup_index],)))
        helpers_test.help_check_equal(simulation.get_positions(index), (old_positions if use_old else new_positions).reshape(count, -1, 3)[lookup_index])
        helpers_test.help_check_equal(simulation.get_velocities(index), (old_velocities if use_old else new_velocities).reshape(count, -1, 3)[lookup_index])

@pytest.mark.parametrize("dump", (False, True))
@pytest.mark.parametrize("load", (None, False, True))
def test_simulation_dump_load_vectors(dump, load):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((100,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 10

    old_vectors = 2 + 3 * numpy.arange(90).reshape(10, 3, 3)
    new_vectors = 5 + 7 * numpy.arange(90).reshape(10, 3, 3)

    simulation.set_vectors(old_vectors)
    data = io.BytesIO()
    simulation.dump(data, vectors=dump)
    simulation.set_vectors(new_vectors)
    data.seek(0)

    if not dump and load:
        with pytest.raises(Exception):
            simulation.load(data, vectors=load)
    else:
        simulation.load(data, vectors=load)
        helpers_test.help_check_equal(simulation.get_vectors(), old_vectors if load or (load is None and dump) else new_vectors)

@pytest.mark.parametrize("dump", (False, True))
@pytest.mark.parametrize("load", (None, False, True))
def test_simulation_dump_load_positions(dump, load):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((100,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 10

    old_positions = 2 + 3 * numpy.arange(3000).reshape(-1, 3)
    new_positions = 5 + 7 * numpy.arange(3000).reshape(-1, 3)

    simulation.set_positions(old_positions)
    data = io.BytesIO()
    simulation.dump(data, positions=dump)
    simulation.set_positions(new_positions)
    data.seek(0)

    if not dump and load:
        with pytest.raises(Exception):
            simulation.load(data, positions=load)
    else:
        simulation.load(data, positions=load)
        helpers_test.help_check_equal(simulation.get_positions(), old_positions if load or (load is None and dump) else new_positions)

@pytest.mark.parametrize("dump", (False, True))
@pytest.mark.parametrize("load", (None, False, True))
def test_simulation_dump_load_velocities(dump, load):
    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((100,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 10

    old_velocities = 2 + 3 * numpy.arange(3000).reshape(-1, 3)
    new_velocities = 5 + 7 * numpy.arange(3000).reshape(-1, 3)

    simulation.set_velocities(old_velocities)
    data = io.BytesIO()
    simulation.dump(data, velocities=dump)
    simulation.set_velocities(new_velocities)
    data.seek(0)

    if not dump and load:
        with pytest.raises(Exception):
            simulation.load(data, velocities=load)
    else:
        simulation.load(data, velocities=load)
        helpers_test.help_check_equal(simulation.get_velocities(), old_velocities if load or (load is None and dump) else new_velocities)

@pytest.mark.parametrize("simulation_precision", multiopenmm.Precision)
@pytest.mark.parametrize("dump_precision", (None, *multiopenmm.Precision))
def test_simulation_dump_load_precision(simulation_precision, dump_precision):
    double_vectors = numpy.ones((10, 3, 3)) + HELP_EPSILON
    double_positions = numpy.ones((1000, 3)) + HELP_EPSILON
    double_velocities = numpy.ones((1000, 3)) + HELP_EPSILON

    single_vectors = double_vectors.astype(numpy.single)
    single_positions = double_positions.astype(numpy.single)
    single_velocities = double_velocities.astype(numpy.single)

    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((100,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), precision=simulation_precision)
    simulation.instance_count = 10
    simulation.set_vectors(double_vectors)
    simulation.set_positions(double_positions)
    simulation.set_velocities(double_velocities)
    data = io.BytesIO()
    simulation.dump(data, precision=dump_precision)

    simulation = multiopenmm.Simulation(helpers_test.help_make_templates((100,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), precision=simulation_precision)
    simulation.instance_count = 10
    data.seek(0)
    simulation.load(data)

    single = simulation_precision == multiopenmm.Precision.SINGLE or dump_precision == multiopenmm.Precision.SINGLE
    helpers_test.help_check_equal(simulation.get_vectors().astype(numpy.double), (single_vectors if single else double_vectors).astype(numpy.double))
    helpers_test.help_check_equal(simulation.get_positions().astype(numpy.double), (single_positions if single else double_positions).astype(numpy.double))
    helpers_test.help_check_equal(simulation.get_velocities().astype(numpy.double), (single_velocities if single else double_velocities).astype(numpy.double))

@pytest.mark.parametrize("positions", (None, object(), 0, 1, 2))
def test_simulation_apply_constraints_positions_type(positions):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.apply_constraints(positions=positions)

@pytest.mark.parametrize("velocities", (None, object(), 0, 1, 2))
def test_simulation_apply_constraints_velocities_type(velocities):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.apply_constraints(velocities=velocities)

@pytest.mark.parametrize("stacks", ((0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (0, 1, 2, 3, 4), (0, 1, 0, 0, 1)))
@pytest.mark.parametrize("positions_option", (False, True))
@pytest.mark.parametrize("velocities_option", (False, True))
@pytest.mark.parametrize("indices_option", ([], [0, 1, 2, 3, 4], [0], [0, 2], [0, 1, 2]))
def test_simulation_apply_constraints(stacks, positions_option, velocities_option, indices_option):
    templates = tuple(helpers_test.help_make_templates((2, 2)))
    template_indices = (0, 0, 1, 1, 1)
    distances = (1.3, 1.7)
    for template, distance in zip(templates, distances, strict=True):
        template.addConstraint(0, 1, distance)
    
    simulation = multiopenmm.Simulation(templates, multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 5
    simulation.set_template_indices(template_indices)
    simulation.set_stacks(stacks)
    simulation.set_property_values("temperature", numpy.linspace(100, 500, 5))
    simulation.set_property_values("constraint_tolerance", 1e-8)

    positions_initial = numpy.array(((-1, 0, 0), (1, 0, 0),) * 5)
    velocities_initial = numpy.array(((-1, 1, 0), (1, -1, 0),) * 5)

    simulation.set_positions(positions_initial)
    simulation.set_velocities(velocities_initial)

    simulation.apply_constraints(positions_option, velocities_option, indices_option)

    positions_final = simulation.get_positions()
    velocities_final = simulation.get_velocities()

    if positions_option:
        for index in range(5):
            if index in indices_option:
                helpers_test.help_check_changed(positions_final[2 * index:2 * (index + 1)], positions_initial[2 * index:2 * (index + 1)])
                assert numpy.linalg.norm(positions_final[2 * index + 1] - positions_final[2 * index]) == pytest.approx(distances[template_indices[index]])
            else:
                helpers_test.help_check_equal(positions_final[2 * index:2 * (index + 1)], positions_initial[2 * index:2 * (index + 1)])
    else:
        helpers_test.help_check_equal(positions_final, positions_initial)

    if velocities_option:
        for index in range(5):
            if index in indices_option:
                helpers_test.help_check_changed(velocities_final[2 * index:2 * (index + 1)], velocities_initial[2 * index:2 * (index + 1)])
                assert velocities_final[2 * index:2 * (index + 1), 0] == pytest.approx(0)
            else:
                helpers_test.help_check_equal(velocities_final[2 * index:2 * (index + 1)], velocities_initial[2 * index:2 * (index + 1)])
    else:
        helpers_test.help_check_equal(velocities_final, velocities_initial)

@pytest.mark.parametrize("tolerance", (object(), 1j))
def test_simulation_minimize_tolerance_type(tolerance):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.minimize(tolerance=tolerance)

@pytest.mark.parametrize("iteration_count", (object(), 1.0))
def test_simulation_minimize_iteration_count_type(iteration_count):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.minimize(iteration_count=iteration_count)

@pytest.mark.parametrize("stacks", ((0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (0, 1, 2, 3, 4), (0, 1, 0, 0, 1)))
@pytest.mark.parametrize("indices_option", ([], [0, 1, 2, 3, 4], [0], [0, 2], [0, 1, 2], [0, 0, 1, 2, 0]))
@pytest.mark.parametrize("tolerance_data", ((None, False), (1e3, False), (1000, False), (1e-8, True)))
@pytest.mark.parametrize("iteration_count_data", ((None, True), (0, True), (1, False), (100, True)))
def test_simulation_minimize(stacks, indices_option, tolerance_data, iteration_count_data):
    tolerance, tolerance_should_minimize = tolerance_data
    iteration_count, iteration_count_should_minimize = iteration_count_data

    templates = tuple(helpers_test.help_make_templates((2, 2)))
    template_indices = (0, 0, 1, 1, 1)
    distances = (1.3, 1.7)
    for template, distance in zip(templates, distances, strict=True):
        force = openmm.HarmonicBondForce()
        force.addBond(0, 1, distance, 1000)
        template.addForce(force)
    
    simulation = multiopenmm.Simulation(templates, multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 5
    simulation.set_template_indices(template_indices)
    simulation.set_stacks(stacks)
    simulation.set_property_values("temperature", numpy.linspace(100, 500, 5))

    positions_initial = numpy.array(((-1, 0, 0), (1, 0, 0),) * 5)
    velocities_initial = numpy.ones((10, 3))

    simulation.set_positions(positions_initial)
    simulation.set_velocities(velocities_initial)

    simulation.minimize(tolerance, iteration_count, indices_option)

    positions_final = simulation.get_positions()
    velocities_final = simulation.get_velocities()

    for index in range(5):
        if index in indices_option:
            if tolerance_should_minimize and iteration_count_should_minimize:
                helpers_test.help_check_changed(positions_final[2 * index:2 * (index + 1)], positions_initial[2 * index:2 * (index + 1)])
                assert numpy.linalg.norm(positions_final[2 * index + 1] - positions_final[2 * index]) == pytest.approx(distances[template_indices[index]])
        else:
            helpers_test.help_check_equal(positions_final[2 * index:2 * (index + 1)], positions_initial[2 * index:2 * (index + 1)])

    helpers_test.help_check_equal(velocities_final, velocities_initial)

@pytest.mark.parametrize("stacks", ((0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (0, 1, 2, 3, 4), (0, 1, 0, 0, 1)))
@pytest.mark.parametrize("indices_option", ([], [0, 1, 2, 3, 4], [0], [0, 2], [0, 1, 2], [0, 0, 1, 2, 0]))
def test_simulation_maxwell_boltzmann(stacks, indices_option):
    templates = tuple(helpers_test.help_make_templates((2, 2)))
    template_indices = (0, 0, 1, 1, 1)

    simulation = multiopenmm.Simulation(templates, multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    simulation.instance_count = 5
    simulation.set_template_indices(template_indices)
    simulation.set_stacks(stacks)
    simulation.set_property_values("temperature", numpy.linspace(100, 500, 5))

    positions_initial = numpy.ones((10, 3))
    velocities_initial = numpy.ones((10, 3))

    simulation.set_positions(positions_initial)
    simulation.set_velocities(velocities_initial)

    simulation.maxwell_boltzmann(indices_option)

    positions_final = simulation.get_positions()
    velocities_final = simulation.get_velocities()

    helpers_test.help_check_equal(positions_final, positions_initial)

    for index in range(5):
        if index in indices_option:
            helpers_test.help_check_changed(velocities_final[2 * index:2 * (index + 1)], velocities_initial[2 * index:2 * (index + 1)])
        else:
            helpers_test.help_check_equal(velocities_final[2 * index:2 * (index + 1)], velocities_initial[2 * index:2 * (index + 1)])

@pytest.mark.parametrize("stacks", ((0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (0, 1, 2, 3, 4), (0, 1, 0, 0, 1)))
@pytest.mark.parametrize("indices", ((), (0, 1, 2, 3, 4), (0,), (0, 2), (0, 1, 2), (0, 0, 1, 2, 0)))
def test_simulation_maxwell_boltzmann_distribution(stacks, indices):
    rng = numpy.random.default_rng((0x333d8d3758be649d, helpers_test.help_deterministic_hash((stacks, indices))))

    template_indices = (0, 0, 1, 1, 1)
    temperatures = rng.uniform(100, 500, len(template_indices))
    beta = 1 / (HELP_K_B * temperatures)
    template_particle_counts = (1000, 2000)
    template_particle_masses = []

    templates = tuple(helpers_test.help_make_templates(template_particle_counts))
    for template, template_particle_count in zip(templates, template_particle_counts, strict=True):
        particle_masses = rng.uniform(1, 2, template_particle_count)
        for particle_index, particle_mass in enumerate(particle_masses):
            template.setParticleMass(particle_index, particle_mass)
        template_particle_masses.append(particle_masses)
    
    simulation = multiopenmm.Simulation(templates, multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), seed=rng)
    simulation.instance_count = len(template_indices)
    simulation.set_template_indices(template_indices)
    simulation.set_stacks(stacks)
    simulation.set_property_values("temperature", temperatures)

    simulation.maxwell_boltzmann(list(indices))

    for index in range(len(template_indices)):
        if index in indices:
            scaled_velocities = simulation.get_velocities(index) * numpy.sqrt(template_particle_masses[template_indices[index]])[:, None]
            for dimension_1 in range(3):
                assert scipy.stats.ks_1samp(scaled_velocities[:, dimension_1],
                    scipy.stats.norm(loc=0, scale=1 / numpy.sqrt(beta[index])).cdf).pvalue >= HELP_P_LIMIT

                for dimension_2 in range(3):
                    if dimension_2 != dimension_1:
                        assert scipy.stats.pearsonr(scaled_velocities[:, dimension_1], scaled_velocities[:, dimension_2]).pvalue >= HELP_P_LIMIT

    for template_index in template_indices:
        for index_1 in range(len(template_indices)):
            if index_1 in indices and template_indices[index_1] == template_index:
                velocities_1 = simulation.get_velocities(index_1)

                for index_2 in range(len(template_indices)):
                    if index_2 in indices and template_indices[index_2] == template_index and index_2 != index_1:
                        velocities_2 = simulation.get_velocities(index_2)

                        for dimension_1 in range(3):
                            for dimension_2 in range(3):
                                assert scipy.stats.pearsonr(velocities_1[:, dimension_1], velocities_2[:, dimension_2]).pvalue >= HELP_P_LIMIT

@pytest.mark.parametrize("stacks", ((0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (0, 1, 2, 3, 4), (0, 1, 0, 0, 1)))
@pytest.mark.parametrize("indices", ((), (0, 1, 2, 3, 4), (0,), (0, 2), (0, 1, 2), (0, 0, 1, 2, 0)))
def test_simulation_evaluate_energies(stacks, indices):
    rng = numpy.random.default_rng((0xf1c2dceb363c7c52, helpers_test.help_deterministic_hash((stacks, indices))))

    template_indices = (0, 0, 1, 1, 1)
    temperatures = rng.uniform(100, 500, len(template_indices))
    beta = 1 / (HELP_K_B * temperatures)
    template_particle_counts = (1000, 2000)
    template_particle_masses = []

    templates = tuple(helpers_test.help_make_templates(template_particle_counts))
    for template, template_particle_count in zip(templates, template_particle_counts, strict=True):
        particle_masses = rng.uniform(1, 2, template_particle_count)
        for particle_index, particle_mass in enumerate(particle_masses):
            template.setParticleMass(particle_index, particle_mass)
        template_particle_masses.append(particle_masses)

        force = openmm.CustomExternalForce("x^2 + y^2 + z^2")
        for particle_index, particle_mass in enumerate(particle_masses):
            force.addParticle(particle_index)
        template.addForce(force)
    
    simulation = multiopenmm.Simulation(templates, multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble(), seed=rng)
    simulation.instance_count = len(template_indices)
    simulation.set_template_indices(template_indices)
    simulation.set_stacks(stacks)
    simulation.set_property_values("temperature", temperatures)

    positions_zero = simulation.get_positions()
    velocities_zero = simulation.get_velocities()
    positions_nonzero = rng.normal(size=positions_zero.shape)
    velocities_nonzero = rng.normal(size=velocities_zero.shape)

    simulation.set_positions(positions_nonzero)
    instance_positions = [simulation.get_positions(index) for index in range(len(template_indices))]

    potential_energies, kinetic_energies = simulation.evaluate_energies(list(indices))
    for index, potential_energy in zip(indices, potential_energies, strict=True):
        assert potential_energy == pytest.approx(numpy.sum(simulation.get_positions(index) ** 2))

    simulation.set_positions(positions_zero)
    simulation.set_velocities(velocities_nonzero)

    potential_energies, kinetic_energies = simulation.evaluate_energies(list(indices))
    for index, potential_energy, kinetic_energy in zip(indices, potential_energies, kinetic_energies, strict=True):
        assert potential_energy == pytest.approx(0)
        assert kinetic_energy == pytest.approx(numpy.sum(template_particle_masses[template_indices[index]][:, None] * simulation.get_velocities(index) ** 2) / 2)

@pytest.mark.parametrize("step_count", (object(), None, 1.0))
def test_simulation_integrate_step_count_type(step_count):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.integrate(step_count)

@pytest.mark.parametrize("step_count", (-10, -1))
def test_simulation_integrate_step_count_value(step_count):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(ValueError):
        simulation.integrate(step_count)

@pytest.mark.parametrize("write_start", (object(), 1.0))
def test_simulation_integrate_write_start_type(write_start):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.integrate(1, write_start=write_start)

@pytest.mark.parametrize("write_start", (-10, -1))
def test_simulation_integrate_write_start_value(write_start):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(ValueError):
        simulation.integrate(1, write_start=write_start)

@pytest.mark.parametrize("write_stop", (object(), 1.0))
def test_simulation_integrate_write_stop_type(write_stop):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.integrate(1, write_stop=write_stop)

@pytest.mark.parametrize("write_stop", (-10, -1))
def test_simulation_integrate_write_stop_value(write_stop):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(ValueError):
        simulation.integrate(1, write_stop=write_stop)

@pytest.mark.parametrize("write_step", (object(), 1.0))
def test_simulation_integrate_write_step_type(write_step):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.integrate(1, write_step=write_step)

@pytest.mark.parametrize("write_step", (-10, -1, 0))
def test_simulation_integrate_write_step_value(write_step):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(ValueError):
        simulation.integrate(1, write_step=write_step)

@pytest.mark.parametrize("write_velocities", (object(), None, 0, 1, 1.0))
def test_simulation_integrate_write_velocities_type(write_velocities):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.integrate(1, write_velocities=write_velocities)

@pytest.mark.parametrize("write_energies", (object(), None, 0, 1, 1.0))
def test_simulation_integrate_write_energies_type(write_energies):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.integrate(1, write_energies=write_energies)

@pytest.mark.parametrize("broadcast_energies", (object(), None, 0, 1, 1.0))
def test_simulation_integrate_broadcast_energies_type(broadcast_energies):
    simulation = multiopenmm.Simulation((), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
    with pytest.raises(TypeError):
        simulation.integrate(1, broadcast_energies=broadcast_energies)

@pytest.mark.parametrize("stacks", ((0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (0, 1, 2, 3, 4), (0, 1, 0, 0, 1)))
@pytest.mark.parametrize("indices", ((), (0, 1, 2, 3, 4), (0,), (0, 2), (0, 1, 2), (0, 0, 1, 2, 0)))
@pytest.mark.parametrize("step_count", (0, 1, 2, 10))
@pytest.mark.parametrize("write_velocities", (False, True))
@pytest.mark.parametrize("write_energies", (False, True))
@pytest.mark.parametrize("broadcast_energies", (False, True))
def test_simulation_integrate_options(stacks, indices, step_count, write_velocities, write_energies, broadcast_energies):
    try:
        with tempfile.TemporaryDirectory() as temporary_path:
            multiopenmm.set_scratch_directory(temporary_path)

            templates = tuple(helpers_test.help_make_templates((2, 3)))
            template_indices = (0, 0, 1, 1, 1)

            simulation = multiopenmm.Simulation(templates, multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
            simulation.instance_count = len(template_indices)
            simulation.set_template_indices(template_indices)
            simulation.set_stacks(stacks)
            simulation.set_property_values("temperature", numpy.linspace(100, 500, 5))

            simulation.set_positions(numpy.linspace(0, 3.8, 39).reshape(13, 3))
            simulation.set_velocities(numpy.linspace(3.9, 7.7, 39).reshape(13, 3))

            positions_initial = [simulation.get_positions(instance_index) for instance_index in range(len(template_indices))]
            velocities_initial = [simulation.get_velocities(instance_index) for instance_index in range(len(template_indices))]

            integration_result = simulation.integrate(step_count, write_start=0, write_stop=step_count + 1, write_step=1,
                write_velocities=write_velocities, write_energies=write_energies, broadcast_energies=broadcast_energies, indices=list(indices))
            if broadcast_energies:
                integration_result, broadcast_result = integration_result

            positions_final = [simulation.get_positions(instance_index) for instance_index in range(len(template_indices))]
            velocities_final = [simulation.get_velocities(instance_index) for instance_index in range(len(template_indices))]

            for index in range(len(template_indices)):
                if index in indices and step_count:
                    helpers_test.help_check_changed(positions_final[index], positions_initial[index])
                    helpers_test.help_check_changed(velocities_final[index], velocities_initial[index])
                else:
                    helpers_test.help_check_equal(positions_final[index], positions_initial[index])
                    helpers_test.help_check_equal(velocities_final[index], velocities_initial[index])

            class TestExporter(multiopenmm.export.Exporter):
                def export(self, get_frames):
                    self.data = tuple(tuple(get_frames(index)) for index in range(len(template_indices)))
            exporter = TestExporter()
            multiopenmm.export_results((integration_result,), (exporter,))
            data = exporter.data

            for index in range(len(template_indices)):
                if index in indices:
                    assert len(data[index]) == step_count + 1
                    for frame in data[index]:
                        assert "vectors" in frame
                        assert "positions" in frame
                        assert ("velocities" in frame) == write_velocities
                        assert ("potential_energy" in frame) == write_energies
                        assert ("kinetic_energy" in frame) == write_energies
                else:
                    assert data[index] == ()

            if broadcast_energies:
                assert broadcast_result.size == len(indices)

    finally:
        multiopenmm.set_scratch_directory()

@pytest.mark.parametrize("step_count", (0, 1, 2, 100))
@pytest.mark.parametrize("write_start", ((0, 1, 2, 3, 19, 99, 100, 101, 137)))
@pytest.mark.parametrize("write_stop", ((0, 1, 2, 3, 19, 99, 100, 101, 137)))
@pytest.mark.parametrize("write_step", ((1, 2, 3, 19, 99, 100, 101, 137)))
def test_simulation_integrate_stepping(step_count, write_start, write_stop, write_step):
    try:
        with tempfile.TemporaryDirectory() as temporary_path:
            multiopenmm.set_scratch_directory(temporary_path)

            write_count = 0
            frame = write_start
            while frame < write_stop:
                if frame <= step_count:
                    write_count += 1
                frame += write_step

            simulation = multiopenmm.Simulation(helpers_test.help_make_templates((1,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
            simulation.instance_count = 1

            integration_result = simulation.integrate(step_count, write_start=write_start, write_stop=write_stop, write_step=write_step)

            class TestExporter(multiopenmm.export.Exporter):
                def export(self, get_frames):
                    self.data = tuple(get_frames(0))
            exporter = TestExporter()
            multiopenmm.export_results((integration_result,), (exporter,))
            data = exporter.data

            assert len(data) == write_count
            for frame in data:
                assert "vectors" in frame
                assert "positions" in frame

    finally:
        multiopenmm.set_scratch_directory()

@pytest.mark.parametrize("step_count", (0, 1, 2, 100))
def test_simulation_integrate_beginning_end(step_count):
    try:
        with tempfile.TemporaryDirectory() as temporary_path:
            multiopenmm.set_scratch_directory(temporary_path)

            simulation = multiopenmm.Simulation(helpers_test.help_make_templates((1,)), multiopenmm.SynchronousManager(), multiopenmm.CanonicalEnsemble())
            simulation.instance_count = 1

            simulation.set_positions(numpy.ones(3))
            simulation.set_velocities(numpy.ones(3))

            initial_positions = simulation.get_positions()
            initial_velocities = simulation.get_velocities()

            integration_result = simulation.integrate(step_count, write_start=0, write_stop=step_count + 1, write_step=max(1, step_count), write_velocities=True)

            final_positions = simulation.get_positions()
            final_velocities = simulation.get_velocities()

            class TestExporter(multiopenmm.export.Exporter):
                def export(self, get_frames):
                    self.data = tuple(get_frames(0))
            exporter = TestExporter()
            multiopenmm.export_results((integration_result,), (exporter,))
            data = exporter.data

            if step_count:
                assert len(data) == 2
                initial_positions_check = data[0]["positions"]
                initial_velocities_check = data[0]["velocities"]
                final_positions_check = data[1]["positions"]
                final_velocities_check = data[1]["velocities"]
            else:
                assert len(data) == 1
                initial_positions_check = data[0]["positions"]
                initial_velocities_check = data[0]["velocities"]
                final_positions_check = data[0]["positions"]
                final_velocities_check = data[0]["velocities"]

            helpers_test.help_check_equal(initial_positions_check, initial_positions)
            helpers_test.help_check_equal(initial_velocities_check, initial_velocities)
            helpers_test.help_check_equal(final_positions_check, final_positions)
            helpers_test.help_check_equal(final_velocities_check, final_velocities)

    finally:
        multiopenmm.set_scratch_directory()

@pytest.mark.xfail
def test_simulation_replica_exchange():
    # TODO: Test Simulation.replica_exchange().
    # TODO: test replica exchange options: *ExchangePairGenerator, *AcceptanceCriterion
    raise NotImplementedError

# TODO: can we change platforms in the midst of a simulation?
# TODO: test exporters: export_results, delete_results, *Exporter
# TODO: test ensembles: Canonical, IsothermalIsobaric, etc., in simulation!
# TODO: stochastic tests with checkensemble or something like this
# TODO: test managers: ThreadPool, ProcessPool, SocketServer
# TODO: basic tutorial in documentation and documentation cleanup
# TODO: are broadcast energies returned correctly?

@pytest.mark.parametrize("thermostat", multiopenmm.Barostat)
def test_canonical_ensemble_init_thermostat_type(thermostat):
    with pytest.raises(TypeError):
        multiopenmm.CanonicalEnsemble(thermostat)

@pytest.mark.parametrize("thermostat", multiopenmm.Thermostat)
def test_canonical_ensemble_thermostat(thermostat):
    ensemble = multiopenmm.CanonicalEnsemble(thermostat)
    assert ensemble.thermostat == thermostat

@pytest.mark.parametrize("thermostat", multiopenmm.Barostat)
@pytest.mark.parametrize("barostat", multiopenmm.Barostat)
def test_isothermal_isobaric_ensemble_init_thermostat_type(thermostat, barostat):
    with pytest.raises(TypeError):
        multiopenmm.IsothermalIsobaricEnsemble(thermostat, barostat)

@pytest.mark.parametrize("thermostat", multiopenmm.Thermostat)
@pytest.mark.parametrize("barostat", multiopenmm.Thermostat)
def test_isothermal_isobaric_ensemble_init_barostat_type(thermostat, barostat):
    with pytest.raises(TypeError):
        multiopenmm.IsothermalIsobaricEnsemble(thermostat, barostat)

@pytest.mark.parametrize("thermostat", multiopenmm.Thermostat)
@pytest.mark.parametrize("barostat", multiopenmm.Barostat)
def test_isothermal_isobaric_ensemble_thermostat_barostat(thermostat, barostat):
    ensemble = multiopenmm.IsothermalIsobaricEnsemble(thermostat, barostat)
    assert ensemble.thermostat == thermostat
    assert ensemble.barostat == barostat
