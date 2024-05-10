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
import os
import pytest
import tempfile

HELP_SCRATCH_NAME = "MULTIOPENMMSCRATCH"

@pytest.mark.parametrize("platform_name", ("A", "ABCDE"))
@pytest.mark.parametrize("platform_properties", (None, {}, {"A": "B"}, {"A": "B", "C": "D"}))
def test_platform_data_init_repr_properties(platform_name, platform_properties):
    platform_data = multiopenmm.PlatformData(platform_name, platform_properties)
    repr(platform_data)
    assert platform_data.platform_name == platform_name
    assert platform_data.platform_properties == ({} if platform_properties is None else platform_properties)

@pytest.mark.parametrize("platform_name", (None, object(), 0))
def test_platform_data_init_platform_name(platform_name):
    with pytest.raises(TypeError):
        multiopenmm.PlatformData(platform_name)

@pytest.mark.parametrize("platform_properties", (object(), ["A", "B"], [("A", "B"), ("C", "D")], {0: "A", "B": "C"}, {"A": "B", "C": 0}))
def test_platform_data_init_platform_properties(platform_properties):
    with pytest.raises(TypeError):
        multiopenmm.PlatformData("A", platform_properties)

@pytest.mark.parametrize("platform_test_data", (
    ("A", {"B": "C", "D": "E"}, "A", {"B": "C", "D": "E"}, True),
    ("A", {"B": "C", "D": "E"}, "a", {"B": "C", "D": "E"}, False),
    ("A", {"B": "C", "D": "E"}, "A", {"b": "C", "D": "E"}, False),
))
def test_platform_data_eq(platform_test_data):
    platform_name_1, platform_properties_1, platform_name_2, platform_properties_2, should_be_equal = platform_test_data
    platform_data_1 = multiopenmm.PlatformData(platform_name_1, platform_properties_1)
    platform_data_2 = multiopenmm.PlatformData(platform_name_2, platform_properties_2)

    assert platform_data_1 == platform_data_1
    assert platform_data_2 == platform_data_2
    assert (platform_data_1 == platform_data_2) == should_be_equal

@pytest.mark.parametrize("manager_type", (multiopenmm.SynchronousManager,))
@pytest.mark.parametrize("platform_data", (None, multiopenmm.PlatformData("A", {"B": "C", "D": "E"})))
def test_manager_init(manager_type, platform_data):
    assert manager_type(platform_data=platform_data).platform_data == platform_data

@pytest.mark.parametrize("manager_type", (multiopenmm.SynchronousManager,))
@pytest.mark.parametrize("platform_data", (object(), "A", {"B": "C", "D": "E"}))
def test_manager_platform_data(manager_type, platform_data):
    with pytest.raises(TypeError):
        manager_type(platform_data=platform_data)

@pytest.mark.parametrize("manager_type", (multiopenmm.SynchronousManager,))
@pytest.mark.parametrize("manager_test_data", (
    (None, multiopenmm.PlatformData("A", {"B": "C", "D": "E"})),
    (multiopenmm.PlatformData("A", {"B": "C", "D": "E"}), None)))
def test_manager_set_platform_data(manager_type, manager_test_data):
    platform_data_1, platform_data_2 = manager_test_data
    manager = manager_type(platform_data_1)
    manager.platform_data = platform_data_2
    assert manager.platform_data == platform_data_2

@pytest.mark.parametrize("args", ((), (1,), (1, 2, 3)))
@pytest.mark.parametrize("kwargs", (dict(), dict(a=1), dict(a=2), dict(b=1), dict(a=1, b=2)))
def test_arguments(args, kwargs):
    arguments = multiopenmm.support.Arguments(*args, **kwargs)
    assert arguments.args == args
    assert arguments.kwargs == kwargs
    def check_apply(*check_args, **check_kwargs):
        assert check_args == args
        assert check_kwargs == check_kwargs
    arguments.apply_to(check_apply)

@pytest.mark.parametrize("test_data", (
    (multiopenmm.support.Arguments(1), 1, False),
    (multiopenmm.support.Arguments(1), multiopenmm.support.Arguments(2), False),
    (multiopenmm.support.Arguments(1), multiopenmm.support.Arguments(1, 2), False),
    (multiopenmm.support.Arguments(1), multiopenmm.support.Arguments(1, a=1), False),
    (multiopenmm.support.Arguments(1, a=1), multiopenmm.support.Arguments(1, a=2), False),
    (multiopenmm.support.Arguments(1, a=1), multiopenmm.support.Arguments(1, b=1), False),
    (multiopenmm.support.Arguments(1, a=1), multiopenmm.support.Arguments(1, a=1, b=2), False),
    (multiopenmm.support.Arguments(1, a=1, b=2), multiopenmm.support.Arguments(1, 2, a=1, b=2), False),
    (multiopenmm.support.Arguments(), multiopenmm.support.Arguments(), True),
    (multiopenmm.support.Arguments(1, 2), multiopenmm.support.Arguments(1, 2), True),
    (multiopenmm.support.Arguments(a=1, b=2), multiopenmm.support.Arguments(a=1, b=2), True),
    (multiopenmm.support.Arguments(1, 2, a=1, b=2), multiopenmm.support.Arguments(1, 2, a=1, b=2), True),
))
def test_arguments_equal(test_data):
    arguments_1, arguments_2, should_be_equal = test_data
    assert (arguments_1 == arguments_2) == should_be_equal

def test_scratch_directory_exists():
    assert os.path.isdir(multiopenmm.get_scratch_directory())

def test_set_scratch_directory():
    try:
        with tempfile.TemporaryDirectory() as temporary_path:
            multiopenmm.set_scratch_directory(temporary_path)
            assert os.path.realpath(temporary_path) == os.path.realpath(multiopenmm.get_scratch_directory())
    finally:
        multiopenmm.set_scratch_directory()

def test_set_scratch_directory_working():
    cwd = os.getcwd()
    has_scratch = HELP_SCRATCH_NAME in os.environ
    scratch_value = os.environ.get(HELP_SCRATCH_NAME)
    try:
        with tempfile.TemporaryDirectory() as temporary_path:
            if has_scratch:
                del os.environ[HELP_SCRATCH_NAME]
            os.chdir(temporary_path)
            multiopenmm.set_scratch_directory()
            assert os.path.realpath(temporary_path) == os.path.realpath(multiopenmm.get_scratch_directory())
    finally:
        os.chdir(cwd)
        if has_scratch:
            os.environ[HELP_SCRATCH_NAME] = scratch_value
        elif HELP_SCRATCH_NAME in os.environ:
            del os.environ[HELP_SCRATCH_NAME]
        multiopenmm.set_scratch_directory()

def test_set_scratch_directory_environ():
    has_scratch = HELP_SCRATCH_NAME in os.environ
    scratch_value = os.environ.get(HELP_SCRATCH_NAME)
    try:
        with tempfile.TemporaryDirectory() as temporary_path:
            os.environ[HELP_SCRATCH_NAME] = temporary_path
            multiopenmm.set_scratch_directory()
            assert os.path.realpath(temporary_path) == os.path.realpath(multiopenmm.get_scratch_directory())
    finally:
        if has_scratch:
            os.environ[HELP_SCRATCH_NAME] = scratch_value
        elif HELP_SCRATCH_NAME in os.environ:
            del os.environ[HELP_SCRATCH_NAME]
        multiopenmm.set_scratch_directory()

def test_raw_io_none():
    file = io.BytesIO()
    multiopenmm.support.RawFileIO.write_header(file)

    file.seek(0)

    multiopenmm.support.RawFileIO.read_header(file)
    with pytest.raises(multiopenmm.MultiOpenMMError):
        multiopenmm.support.RawFileIO.read_frame(file)

@pytest.mark.parametrize("particle_count", (0, 1, 2, 100))
@pytest.mark.parametrize("write_vectors", (0, 1))
@pytest.mark.parametrize("write_positions", (0, 1))
@pytest.mark.parametrize("write_energy", (0, 1))
def test_raw_io_single(particle_count, write_vectors, write_positions, write_energy):
    rng = numpy.random.default_rng((0xd44b0a917e3d356e, helpers_test.help_deterministic_hash((particle_count, write_vectors, write_positions, write_energy))))

    vectors = rng.normal(size=(3, 3)) if write_vectors else None
    positions = rng.normal(size=(particle_count, 3)) if write_positions else None
    energy = rng.normal() if write_energy else None

    file = io.BytesIO()
    multiopenmm.support.RawFileIO.write_header(file)
    multiopenmm.support.RawFileIO.write_frame(file, vectors, positions, energy)

    file.seek(0)

    multiopenmm.support.RawFileIO.read_header(file)
    read_vectors, read_positions, read_energy = multiopenmm.support.RawFileIO.read_frame(file)
    helpers_test.help_check_equal_none(vectors, read_vectors)
    helpers_test.help_check_equal_none(positions, read_positions)
    assert energy == read_energy
    with pytest.raises(multiopenmm.MultiOpenMMError):
        multiopenmm.support.RawFileIO.read_frame(file)

@pytest.mark.parametrize("frame_count", (2, 1000))
@pytest.mark.parametrize("particle_count_kind", (0, 1, 2, 100, None))
@pytest.mark.parametrize("write_vectors", (0, 1, 2))
@pytest.mark.parametrize("write_positions", (0, 1, 2))
@pytest.mark.parametrize("write_energy", (0, 1, 2))
def test_raw_io_multiple(frame_count, particle_count_kind, write_vectors, write_positions, write_energy):
    rng = numpy.random.default_rng((0x1b44f9b2f2f66147, helpers_test.help_deterministic_hash((frame_count, particle_count_kind, write_vectors, write_positions, write_energy))))

    written_vectors = []
    written_positions = []
    written_energy = []
    
    file = io.BytesIO()
    multiopenmm.support.RawFileIO.write_header(file)
    for frame_index in range(frame_count):
        particle_count = (rng.integers(2) if rng.integers(2) else rng.integers(2, 101)) if particle_count_kind is None else particle_count_kind
        vectors = rng.normal(size=(3, 3)) if write_vectors == 2 or (write_vectors == 1 and rng.integers(2)) else None
        positions = rng.normal(size=(particle_count, 3)) if write_positions == 2 or (write_positions == 1 and rng.integers(2)) else None
        energy = rng.normal() if write_energy == 2 or (write_energy == 1 and rng.integers(2)) else None

        multiopenmm.support.RawFileIO.write_frame(file, vectors, positions, energy)
        written_vectors.append(vectors)
        written_positions.append(positions)
        written_energy.append(energy)
    
    file.seek(0)
    multiopenmm.support.RawFileIO.read_header(file)
    for frame_index in range(frame_count):
        read_vectors, read_positions, read_energy = multiopenmm.support.RawFileIO.read_frame(file)
        helpers_test.help_check_equal_none(written_vectors[frame_index], read_vectors)
        helpers_test.help_check_equal_none(written_positions[frame_index], read_positions)
        assert written_energy[frame_index] == read_energy
    with pytest.raises(multiopenmm.MultiOpenMMError):
        multiopenmm.support.RawFileIO.read_frame(file)

@pytest.mark.parametrize("result_count", (0, 1, 2, 100))
def test_result_io(result_count):
    rng = numpy.random.default_rng((0x1b44f9b2f2f66147, helpers_test.help_deterministic_hash((result_count,))))

    written_results = []

    file = io.BytesIO()
    multiopenmm.support.ResultFileIO.write_header(file)
    for result_index in range(result_count):
        result = multiopenmm.IntegrationResult(*map(tuple, rng.integers(1 << 30, size=(7, 100))))
        multiopenmm.support.ResultFileIO.write_result(file, result)
        written_results.append(result)

    file.seek(0)
    multiopenmm.support.ResultFileIO.read_header(file)
    for result_index in range(result_count):
        read_result = multiopenmm.support.ResultFileIO.read_result(file)
        assert written_results[result_index] == read_result
    with pytest.raises(multiopenmm.MultiOpenMMError):
        multiopenmm.support.ResultFileIO.read_result(file)
