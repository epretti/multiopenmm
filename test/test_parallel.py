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

@pytest.mark.xfail
def test_simulation_apply_constraints():
    # TODO: Test Simulation.apply_constraints().
    raise NotImplementedError

@pytest.mark.xfail
def test_simulation_minimize():
    # TODO: Test Simulation.minimize().
    raise NotImplementedError

@pytest.mark.xfail
def test_simulation_maxwell_boltzmann():
    # TODO: Test Simulation.maxwell_boltzmann().
    raise NotImplementedError

@pytest.mark.xfail
def test_simulation_evaluate_energies():
    # TODO: Test Simulation.evaluate_energies().
    raise NotImplementedError

@pytest.mark.xfail
def test_simulation_integrate():
    # TODO: Test Simulation.integrate().
    raise NotImplementedError

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
