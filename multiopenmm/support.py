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

import os

class MultiOpenMMError(Exception):
    """
    Raised when a MultiOpenMM-specific error condition arises.
    """

    __slots__ = ()

class MultiOpenMMWarning(Warning):
    """
    Issued when a MultiOpenMM-specific warning condition arises.
    """

    __slots__ = ()

class Arguments:
    # Holds function call arguments.

    __slots__ = ("__args", "__kwargs")

    def __init__(self, /, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    @property
    def args(self):
        return self.__args

    @property
    def kwargs(self):
        return self.__kwargs

    def apply_to(self, function):
        # Calls a given function with the stored arguments.

        return function(*self.__args, **self.__kwargs)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return (self.__args, self.__kwargs) == (other.__args, other.__kwargs)

def get_scratch_directory():
    """
    Retrieves the scratch directory used by MultiOpenMM for temporary files.

    Returns
    -------
    str
        The absolute path to the current scratch directory.
    """

    return os.path.realpath(__scratch_directory)

def set_scratch_directory(path=None):
    """
    Sets the scratch directory used by MultiOpenMM for temporary files.  By
    default, this will be the current working directory, unless overridden by
    setting the environment variable ``MULTIOPENMMSCRATCH``.  The directory must
    exist and be writable or errors will be raised when MultiOpenMM attempts to
    create temporary files.

    Parameters
    ----------
    path : str, optional
        A path to a directory in which MultiOpenMM should create temporary
        files.  If none is provided, the default behavior described will be
        restored.
    """

    global __scratch_directory
    
    if path is None:
        __scratch_directory = os.getenv("MULTIOPENMMSCRATCH", ".")

    else:
        if not isinstance(path, str):
            raise TypeError("path must be a str")

        __scratch_directory = path

# Initialize __scratch_directory.
set_scratch_directory()

def get_seed(rng):
    # Returns a positive random seed that will not overflow a 32-bit
    # integer.  This explicitly excludes zero, as this is sometimes interpreted
    # by OpenMM as an instruction to choose a seed non-deterministically.

    return rng.integers(1, 0x80000000)
