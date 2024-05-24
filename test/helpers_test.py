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

import hashlib
import numpy
import openmm
import pickle

def help_check_equal(array_1, array_2):
    assert array_1.shape == array_2.shape
    assert numpy.all(array_1 == array_2)

def help_check_changed(array_1, array_2):
    assert array_1.shape == array_2.shape
    assert numpy.any(array_1 != array_2)

def help_check_equal_none(array_1, array_2):
    if array_1 is None and array_2 is None:
        return
    assert array_1 is not None
    assert array_2 is not None
    help_check_equal(array_1, array_2)

def help_make_templates(template_sizes=None, *, override=None):
    if template_sizes is None:
        return override

    for template_size in template_sizes:
        template = openmm.System()
        for particle_index in range(template_size):
            template.addParticle(1.0)
        yield template

def help_deterministic_hash(tuples):
    return int.from_bytes(hashlib.sha3_512(pickle.dumps(tuples)).digest(), byteorder="little")
