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

import bz2
import hashlib
import helpers_test
import inspect
import itertools
import math
import multiopenmm
import numpy
import openmm
import os
import pickle
import pytest
import sys
import warnings

HELP_TEMPLATE_SIZES = (
    (),
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (5,),
    (100,),
    (0, 0),
    (0, 1),
    (0, 100),
    (1, 1),
    (1, 100),
    (100, 100),
    (100, 200),
    (100, 200, 300),
    (100, 200, 100, 100, 300, 0, 400, 0, 0, 500, 600),
)

HELP_TEMPLATE_INDICES = (
    (),
    (0,),
    (0, 0),
    (0, 0, 0),
    (1,),
    (0, 1),
    (1, 1, 0, 0, 1, 0),
    (2, 1, 0),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    (9, 9, 3, 3, 5, 5, 5, 4, 5, 9, 4, 3, 5, 9, 3),
)

HELP_TEMPLATE_SIZES_INDICES = tuple(
    (sizes, indices)
    for sizes in HELP_TEMPLATE_SIZES
    for indices in HELP_TEMPLATE_INDICES
    if all(index < len(sizes) for index in indices)
)

HELP_TEMPLATE_SIZES_INDICES_USED_UNUSED = tuple(
    (sizes, indices)
    for sizes, indices in HELP_TEMPLATE_SIZES_INDICES
    if len(set(indices)) > 1
)

HELP_TEMPLATE_SIZES_INDICES_PARTICLES = tuple(
    (sizes, indices)
    for sizes, indices in HELP_TEMPLATE_SIZES_INDICES
    if any(sizes[index] for index in indices)
)

HELP_PBC_CLASS_DATA_LIST = (((False,),), ((True,),), ((False,), (True,)), ((False,), (True,), (False,), (False,), (True,)))

HELP_CMAP_CLASS_DATA_LIST = (
    ((False, (5,)),),
    ((True, (5,)),),
    ((False, (5,)), (True, (5,))),
    ((False, (5,)), (True, (5,)), (False, (5,)), (False, (5,)), (True, (5,))),
    ((False, (5, 6)),),
    ((False, (5, 6)), (False, (5, 7))),
    ((False, (5, 6)), (False, (7, 8, 9))),
    ((False, (5, 6)), (False, (5, 7)), (True, (5, 6)), (False, (5, 6))),
)

HELP_CUSTOM_EXTERNAL_CLASS_DATA_LIST = (
    (("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb")),),
    (("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb")), ("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb"))),
    (("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb")), ("scale*x*x+scalea*y*y-scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb"))),
    (("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb")), ("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0), ("scalec", 3.0)), ("scalea", "scaleb"))),
    (("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0), ("scalec", 3.0)), ("scalea", "scaleb")), ("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0), ("scaled", 3.0)), ("scalea", "scaleb"))),
    (("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb")), ("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb", "scalec"))),
    (("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb", "scalec")), ("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb", "scaled"))),
    (("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb")), ("scale*x*x+scalea*y*y-scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb")), ("scale*x*x+scalea*y*y+scaleb*z*z", (("scale", 2.0),), ("scalea", "scaleb"))),
)

HELP_CUSTOM_BOND_CLASS_DATA_LIST = (
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",)),),
    (("scale*r*(r+scalea)+scaleb*r*r", False, (("scale", 2.0, True), ("scalea", 3.0, True), ("scaleb", 5.0, True)), ()), ("scale*r*(r+scalec)+scaled*r*r", False, (("scale", 2.0, True), ("scalec", 7.0, True), ("scaled", 11.0, True)), ())),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",))),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*r*(r-scalea)", False, (("scale", 2.0, True),), ("scalea",))),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*r*(r+scalea)", True, (("scale", 2.0, True),), ("scalea",))),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*r*(r+scalea)", False, (("scale", 2.0, True), ("scaleb", 3.0, True)), ("scalea",))),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True), ("scaleb", 3.0, True)), ("scalea",)), ("scale*r*(r+scalea)", False, (("scale", 2.0, True), ("scalec", 3.0, True)), ("scalea",))),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*r*(r+scalea)", False, (("scale", 2.0, False),), ("scalea",))),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea", "scaleb"))),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea", "scaleb")), ("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea", "scalec"))),
    (("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*r*(r-scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*r*(r+scalea)", False, (("scale", 2.0, True),), ("scalea",))),
)

HELP_CUSTOM_ANGLE_TORSION_CLASS_DATA_LIST = (
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",)),),
    (("scale*theta*(theta+scalea)+scaleb*theta*theta", False, (("scale", 2.0, True), ("scalea", 3.0, True), ("scaleb", 5.0, True)), ()), ("scale*theta*(theta+scalec)+scaled*theta*theta", False, (("scale", 2.0, True), ("scalec", 7.0, True), ("scaled", 11.0, True)), ())),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",))),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*theta*(theta-scalea)", False, (("scale", 2.0, True),), ("scalea",))),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*theta*(theta+scalea)", True, (("scale", 2.0, True),), ("scalea",))),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*theta*(theta+scalea)", False, (("scale", 2.0, True), ("scaleb", 3.0, True)), ("scalea",))),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True), ("scaleb", 3.0, True)), ("scalea",)), ("scale*theta*(theta+scalea)", False, (("scale", 2.0, True), ("scalec", 3.0, True)), ("scalea",))),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*theta*(theta+scalea)", False, (("scale", 2.0, False),), ("scalea",))),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea", "scaleb"))),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea", "scaleb")), ("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea", "scalec"))),
    (("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*theta*(theta-scalea)", False, (("scale", 2.0, True),), ("scalea",)), ("scale*theta*(theta+scalea)", False, (("scale", 2.0, True),), ("scalea",))),
)

HELP_CUSTOM_COMPOUND_CLASS_DATA_LIST = (
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (5, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2-scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", True, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True), ("scaleb", 3.0, True)), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True), ("scaleb", 2.0, True)), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True), ("scalec", 3.0, True)), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, False),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea", "scaleb"),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea", "scaleb"),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea", "scalec"),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),
                ("scalec", openmm.Continuous1DFunction, (numpy.cos(numpy.arange(10)), -numpy.pi, numpy.pi, False)))),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),
                ("scaled", openmm.Continuous1DFunction, (numpy.cos(numpy.arange(10)), -numpy.pi, numpy.pi, False)))),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(abs(q3));q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(4)), 0, 4, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(abs(q3));q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Discrete1DFunction, (numpy.sin(numpy.arange(4)),)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -2 * numpy.pi, numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, 2 * numpy.pi, False)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10) % 9), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10) % 9), -numpy.pi, numpy.pi, True)),)),
    ),
    (
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2-scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
        (4, "scale*q1+scalea*q2+scaleb(q3);q1=angle(p1,p2,p3);q2=angle(p2,p3,p4);q3=dihedral(p1,p2,p3,p4)", False, (("scale", 2.0, True),), ("scalea",),
            (("scaleb", openmm.Continuous1DFunction, (numpy.sin(numpy.arange(10)), -numpy.pi, numpy.pi, False)),)),
    ),
)

HELP_CUSTOM_NONBONDED_CLASS_DATA_LIST = (
    (
        ("(1+scale(r))/(r*r+(scaleb+scalec1)*(scaleb+scalec2))", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (), (("scaleb", 0.5, True),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(2+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.CutoffNonPeriodic, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, False, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, True, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 4, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 1, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 1, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False), ("scaled", 1.0, False)), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False), ("scaled", 1.0, False)), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 1, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False), ("scalee", 2.0, False)), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+(scaleb+scalec1)*(scaleb+scalec2))", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (), (("scaleb", 0.5, True),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+(scaleb+scalec1)*(scaleb+scalec2))", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec", "scaled"),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec", "scaled"),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec", "scalee"),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),
                ("scaled", openmm.Continuous1DFunction, (2 + numpy.cos(numpy.linspace(0, 1, 10)), 0, 3, False)))),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
    (
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(2+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
        ("(1+scale(r))/(r*r+scalea1*scalea2)", openmm.CustomNonbondedForce.NoCutoff, True, False, 3, 2, (("scalea", "scaleb+scalec"),), (("scaleb", 0.5, False),), ("scalec",),
            (("scale", openmm.Continuous1DFunction, (2 + numpy.sin(numpy.linspace(0, 1, 10)), 0, 3, False)),)),
    ),
)

with bz2.open(os.path.join(os.path.dirname(os.path.realpath(inspect.getsourcefile(inspect.currentframe()))), "sample_systems.pickle.bz2"), "rb") as help_sample_file:
    HELP_SAMPLE_SYSTEMS, HELP_SAMPLE_POSITIONS = zip(*pickle.load(help_sample_file))

HELP_TEMPLATE_INDICES_SAMPLE = (
    (0,),
    (0, 0),
    (0, 0, 0),
    (1,),
    (0, 1),
    (1, 1, 0, 0, 1, 0),
    (2, 1, 0),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (9, 9, 3, 3, 5, 5, 5, 4, 5, 9, 4, 3, 5, 9, 3),
)

HELP_FORCE_GROUP_CLASS_DATA_LIST = ((0,), (1,), (0, 1), (2,), (0, 2), (1, 3, 5), (0, 0, 1, 2), (1, 3, 5, 3, 3, 7, 5, 3, 4, 4, 2))

def help_get_uniform_sum(rng, count, limit):
    results = []
    for index in range(count - 1):
        results.append((limit - math.fsum(results)) * (1 - rng.uniform() ** (1 / (count - index - 1))))
    results.append(limit - math.fsum(results))
    return numpy.array(results)

def help_get_zero_sum(rng, count):
    results = rng.normal(size=count)
    return results - math.fsum(results) / count

def help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, reference_positions, force_group=None):
    reference_energies = numpy.zeros(len(template_indices))
    reference_forces = numpy.zeros((particle_offsets[-1], 3))

    reference_derivatives = [{} for instance_index in range(len(template_indices))]
    derivatives_wanted = set()

    platform = openmm.Platform.getPlatformByName("Reference")

    for template_index, template in enumerate(templates):
        if template_index not in template_indices:
            continue

        if template.getNumParticles():
            context = openmm.Context(template, openmm.VerletIntegrator(1), platform)

            for instance_index, target_template_index in enumerate(template_indices):
                if template_index != target_template_index:
                    continue

                particle_index_1 = particle_offsets[instance_index]
                particle_index_2 = particle_offsets[instance_index + 1]
                temperature_scale = temperature_scales[instance_index]

                context.setPositions(reference_positions[particle_index_1:particle_index_2])
                if force_group is None:
                    state = context.getState(getForces=True, getEnergy=True, getParameterDerivatives=True)
                else:
                    state = context.getState(getForces=True, getEnergy=True, getParameterDerivatives=True, groups=1 << force_group)

                reference_energies[instance_index] = state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system) / temperature_scale
                reference_forces[particle_index_1:particle_index_2] = state.getForces(asNumpy=True).value_in_unit_system(openmm.unit.md_unit_system) / temperature_scale

                derivatives = {name: value / temperature_scale for name, value in dict(state.getEnergyParameterDerivatives()).items()}
                reference_derivatives[instance_index] = derivatives
                derivatives_wanted.update(derivatives)
        else:
            for instance_index, target_template_index in enumerate(template_indices):
                if template_index != target_template_index:
                    continue

                reference_energies[instance_index] = 0

    context = openmm.Context(combined_system, openmm.VerletIntegrator(1), platform)
    context.setPositions(reference_positions)
    if force_group is None:
        state = context.getState(getForces=True, getEnergy=True, getParameterDerivatives=True)
    else:
        state = context.getState(getForces=True, getEnergy=True, getParameterDerivatives=True, groups=1 << force_group)
    
    assert state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system) == pytest.approx(numpy.sum(reference_energies))
    assert state.getForces(asNumpy=True).value_in_unit_system(openmm.unit.md_unit_system) == pytest.approx(reference_forces)

    derivatives = dict(state.getEnergyParameterDerivatives())
    assert set(derivatives) == derivatives_wanted
    for derivative in sorted(derivatives_wanted):
        assert derivatives[derivative] == pytest.approx(sum(instance_derivatives.get(derivative, 0) for instance_derivatives in reference_derivatives))

def help_deterministic_hash(tuples):
    return int.from_bytes(hashlib.sha3_512(pickle.dumps(tuples)).digest(), byteorder="little")

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES)
@pytest.mark.filterwarnings("ignore:.*No templates in use.*:multiopenmm.MultiOpenMMWarning")
def test_stack(template_data):
    template_sizes, template_indices = template_data
    instance_count = len(template_indices)
    instance_sizes = tuple(template_sizes[template_index] for template_index in template_indices)

    combined_system, particle_offsets = multiopenmm.stacking.stack(helpers_test.help_make_templates(template_sizes), template_indices, numpy.ones(instance_count))
    assert combined_system.getNumParticles() == sum(instance_sizes)
    helpers_test.help_check_equal(particle_offsets, numpy.cumsum((0,) + instance_sizes))

@pytest.mark.parametrize("template_sizes", HELP_TEMPLATE_SIZES)
@pytest.mark.filterwarnings("ignore:.*No templates in use.*:multiopenmm.MultiOpenMMWarning")
def test_stack_iterable(template_sizes):
    template_count = len(template_sizes)
    combined_system, particle_offsets = multiopenmm.stacking.stack(helpers_test.help_make_templates(template_sizes), numpy.arange(template_count), numpy.ones(template_count))
    assert combined_system.getNumParticles() == sum(template_sizes)
    helpers_test.help_check_equal(particle_offsets, numpy.cumsum((0,) + template_sizes))

@pytest.mark.parametrize("template_sizes", HELP_TEMPLATE_SIZES)
@pytest.mark.filterwarnings("ignore:.*No templates in use.*:multiopenmm.MultiOpenMMWarning")
def test_stack_tuple(template_sizes):
    template_count = len(template_sizes)
    combined_system, particle_offsets = multiopenmm.stacking.stack(tuple(helpers_test.help_make_templates(template_sizes)), numpy.arange(template_count), numpy.ones(template_count))
    assert combined_system.getNumParticles() == sum(template_sizes)
    helpers_test.help_check_equal(particle_offsets, numpy.cumsum((0,) + template_sizes))

@pytest.mark.parametrize("template_sizes", HELP_TEMPLATE_SIZES)
@pytest.mark.filterwarnings("ignore:.*No templates in use.*:multiopenmm.MultiOpenMMWarning")
def test_stack_list(template_sizes):
    template_count = len(template_sizes)
    combined_system, particle_offsets = multiopenmm.stacking.stack(list(helpers_test.help_make_templates(template_sizes)), numpy.arange(template_count), numpy.ones(template_count))
    assert combined_system.getNumParticles() == sum(template_sizes)
    helpers_test.help_check_equal(particle_offsets, numpy.cumsum((0,) + template_sizes))

@pytest.mark.parametrize("templates", (None, object(), openmm.System(), (openmm.System(), None)))
def test_stack_templates_type(templates):
    with pytest.raises(TypeError):
        multiopenmm.stacking.stack(templates, (), ())

@pytest.mark.parametrize("template_indices_shape", ((1, 1), (3, 1), (1, 3), (1, 0), (2, 3, 4)))
def test_stack_template_indices_ndim(template_indices_shape):
    with pytest.raises(ValueError):
        multiopenmm.stacking.stack(helpers_test.help_make_templates((1,)), numpy.zeros(template_indices_shape, dtype=int), numpy.ones(template_indices_shape))

@pytest.mark.parametrize("template_indices_range_data", ((0, (0,)), (1, (0, 0, 1, 0)), (2, (0, -1, 1))))
def test_stack_template_indices_range(template_indices_range_data):
    template_count, template_indices = template_indices_range_data
    with pytest.raises(ValueError):
        multiopenmm.stacking.stack(helpers_test.help_make_templates((1,) * template_count), template_indices, numpy.ones_like(template_indices))

@pytest.mark.parametrize("temperature_scales_shape", ((2, 0), (0, 2), (2, 2), (2, 3, 4)))
def test_stack_temperature_scales_ndim(temperature_scales_shape):
    with pytest.raises(ValueError):
        multiopenmm.stacking.stack(helpers_test.help_make_templates((1,)), numpy.zeros(2, dtype=int), numpy.ones(temperature_scales_shape))

@pytest.mark.parametrize("temperature_scales_size_data", ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 3)))
def test_stack_temperature_scales_size(temperature_scales_size_data):
    template_index_count, temperature_scale_count = temperature_scales_size_data
    with pytest.raises(ValueError):
        multiopenmm.stacking.stack(helpers_test.help_make_templates((1,)), numpy.zeros(template_index_count, dtype=int), numpy.ones(temperature_scale_count))

@pytest.mark.parametrize("template_sizes", HELP_TEMPLATE_SIZES)
def test_stack_vectors_none(template_sizes):
    with pytest.warns(multiopenmm.MultiOpenMMWarning, match=".*No templates in use.*"):
        multiopenmm.stacking.stack(helpers_test.help_make_templates(template_sizes), (), ())

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_USED_UNUSED)
def test_stack_vectors_mismatch(template_data):
    template_sizes, template_indices = template_data
    used_indices = sorted(set(template_indices))
    unused_indices = sorted(set(range(len(template_sizes))) - set(template_indices))

    def modify_templates_list(templates_list, index_list):
        template_index = index_list[len(index_list) // 2]
        a, b, c = templates_list[template_index].getDefaultPeriodicBoxVectors()
        b *= 3
        templates_list[template_index].setDefaultPeriodicBoxVectors(a, b, c)
        return (a, b, c)

    if used_indices:
        templates = list(helpers_test.help_make_templates(template_sizes))
        modify_templates_list(templates, used_indices)
        with pytest.warns(multiopenmm.MultiOpenMMWarning, match=".*Mismatched default periodic box vectors.*"):
            combined_system = multiopenmm.stacking.stack(templates, template_indices, numpy.ones(len(template_indices)))[0]

    if unused_indices:
        templates = list(helpers_test.help_make_templates(template_sizes))
        modify_templates_list(templates, unused_indices)
        with warnings.catch_warnings():
            combined_system = multiopenmm.stacking.stack(templates, template_indices, numpy.ones(len(template_indices)))[0]

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
def test_stack_particles(template_data):
    rng = numpy.random.default_rng((0xd24be100306bbf39, help_deterministic_hash(template_data)))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))

    for template_index, template in enumerate(templates):
        for particle_index, mass in enumerate(numpy.exp(rng.uniform(-2, 2, template.getNumParticles()))):
            template.setParticleMass(particle_index, mass)

    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)

    for instance_index, template_index in enumerate(template_indices):
        particle_offset = particle_offsets[instance_index]
        for particle_index in range(template_sizes[template_index]):
            combined_mass = combined_system.getParticleMass(particle_offset + particle_index).value_in_unit_system(openmm.unit.md_unit_system)
            reference_mass = templates[template_index].getParticleMass(particle_index).value_in_unit_system(openmm.unit.md_unit_system)
            assert combined_mass == pytest.approx(reference_mass / temperature_scales[instance_index])

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
def test_stack_constraints(template_data):
    rng = numpy.random.default_rng((0xaf4d11387d01de23, help_deterministic_hash(template_data)))

    template_sizes, template_indices = template_data
    templates = tuple(helpers_test.help_make_templates(template_sizes))

    for template_index, template in enumerate(templates):
        pairs = tuple(itertools.combinations(range(template.getNumParticles()), 2))
        for pair_index in rng.choice(len(pairs), max(min(len(pairs), 10), len(pairs) // 10), replace=False):
            template.addConstraint(*pairs[pair_index], numpy.exp(rng.uniform(-2, 2)))

    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, numpy.ones_like(template_indices))
    constraints = {(particle_index_1, particle_index_2): distance for particle_index_1, particle_index_2, distance in (
        combined_system.getConstraintParameters(constraint_index) for constraint_index in range(combined_system.getNumConstraints()))}

    for instance_index, template_index in enumerate(template_indices):
        particle_offset = particle_offsets[instance_index]
        for constraint_index in range(templates[template_index].getNumConstraints()):
            particle_index_1, particle_index_2, distance = templates[template_index].getConstraintParameters(constraint_index)
            assert constraints.pop((particle_index_1 + particle_offset, particle_index_2 + particle_offset)) == distance

    assert not constraints

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
def test_stack_virtual_sites(template_data):
    rng = numpy.random.default_rng((0xaa985a186f5f6ec9, help_deterministic_hash(template_data)))

    template_sizes, template_indices = template_data
    templates = tuple(helpers_test.help_make_templates(template_sizes))

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        virtual_site_indices = rng.choice(particle_count, max(min(particle_count // 2, 10), particle_count // 10), replace=False)
        non_virtual_site_indices = numpy.array(sorted(set(range(particle_count)) - set(virtual_site_indices)))

        if len(non_virtual_site_indices) < 2:
            continue
        for particle_index in virtual_site_indices:
            template.setVirtualSite(particle_index, openmm.TwoParticleAverageSite(*rng.choice(non_virtual_site_indices, 2, replace=False), 0.5, 0.5))

    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, numpy.ones_like(template_indices))
    virtual_sites = {particle_index: combined_system.getVirtualSite(particle_index)
        for particle_index in range(combined_system.getNumParticles()) if combined_system.isVirtualSite(particle_index)}

    for instance_index, template_index in enumerate(template_indices):
        particle_offset = particle_offsets[instance_index]
        for particle_index in range(template_sizes[template_index]):
            if templates[template_index].isVirtualSite(particle_index):
                virtual_sites.pop(particle_index + particle_offset)
            else:
                assert particle_index + particle_offset not in virtual_sites

    assert not virtual_sites
    
@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
def test_two_particle_average_site(template_data):
    rng = numpy.random.default_rng((0x4cf82905e9f42d53, help_deterministic_hash(template_data)))

    template_sizes, template_indices = template_data
    templates = tuple(helpers_test.help_make_templates(template_sizes))

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 3:
            continue
        indices = rng.choice(particle_count, 3 * rng.integers(max(1, particle_count // 12), max(2, particle_count // 4)), replace=False).reshape(-1, 3)

        for virtual_site_index, particle_index_1, particle_index_2 in indices:
            weight_1 = rng.uniform(0, 1)
            template.setVirtualSite(virtual_site_index, openmm.TwoParticleAverageSite(
                particle_index_1,
                particle_index_2,
                *help_get_uniform_sum(rng, 2, 1),
            ))

    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, numpy.ones_like(template_indices))
    virtual_sites = {particle_index: combined_system.getVirtualSite(particle_index)
        for particle_index in range(combined_system.getNumParticles()) if combined_system.isVirtualSite(particle_index)}

    for instance_index, template_index in enumerate(template_indices):
        particle_offset = particle_offsets[instance_index]
        for particle_index in range(template_sizes[template_index]):
            if templates[template_index].isVirtualSite(particle_index):
                combined_virtual_site = virtual_sites.pop(particle_index + particle_offset)
                reference_virtual_site = templates[template_index].getVirtualSite(particle_index)
                assert isinstance(combined_virtual_site, openmm.TwoParticleAverageSite)
                assert combined_virtual_site.getParticle(0) == reference_virtual_site.getParticle(0) + particle_offset
                assert combined_virtual_site.getParticle(1) == reference_virtual_site.getParticle(1) + particle_offset
                assert combined_virtual_site.getWeight(0) == reference_virtual_site.getWeight(0)
                assert combined_virtual_site.getWeight(1) == reference_virtual_site.getWeight(1)
            else:
                assert particle_index + particle_offset not in virtual_sites

    assert not virtual_sites

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
def test_three_particle_average_site(template_data):
    rng = numpy.random.default_rng((0x79b39c159ecde9fb, help_deterministic_hash(template_data)))

    template_sizes, template_indices = template_data
    templates = tuple(helpers_test.help_make_templates(template_sizes))

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 4:
            continue
        indices = rng.choice(particle_count, 4 * rng.integers(max(1, particle_count // 16), max(2, 3 * particle_count // 16)), replace=False).reshape(-1, 4)

        for virtual_site_index, particle_index_1, particle_index_2, particle_index_3 in indices:
            template.setVirtualSite(virtual_site_index, openmm.ThreeParticleAverageSite(
                particle_index_1,
                particle_index_2,
                particle_index_3,
                *help_get_uniform_sum(rng, 3, 1),
            ))

    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, numpy.ones_like(template_indices))
    virtual_sites = {particle_index: combined_system.getVirtualSite(particle_index)
        for particle_index in range(combined_system.getNumParticles()) if combined_system.isVirtualSite(particle_index)}

    for instance_index, template_index in enumerate(template_indices):
        particle_offset = particle_offsets[instance_index]
        for particle_index in range(template_sizes[template_index]):
            if templates[template_index].isVirtualSite(particle_index):
                combined_virtual_site = virtual_sites.pop(particle_index + particle_offset)
                reference_virtual_site = templates[template_index].getVirtualSite(particle_index)
                assert isinstance(combined_virtual_site, openmm.ThreeParticleAverageSite)
                assert combined_virtual_site.getParticle(0) == reference_virtual_site.getParticle(0) + particle_offset
                assert combined_virtual_site.getParticle(1) == reference_virtual_site.getParticle(1) + particle_offset
                assert combined_virtual_site.getParticle(2) == reference_virtual_site.getParticle(2) + particle_offset
                assert combined_virtual_site.getWeight(0) == reference_virtual_site.getWeight(0)
                assert combined_virtual_site.getWeight(1) == reference_virtual_site.getWeight(1)
                assert combined_virtual_site.getWeight(2) == reference_virtual_site.getWeight(2)
            else:
                assert particle_index + particle_offset not in virtual_sites

    assert not virtual_sites

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
def test_out_of_plane_site(template_data):
    rng = numpy.random.default_rng((0x2cc7b4bf9ab3f3b2, help_deterministic_hash(template_data)))

    template_sizes, template_indices = template_data
    templates = tuple(helpers_test.help_make_templates(template_sizes))

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 4:
            continue
        indices = rng.choice(particle_count, 4 * rng.integers(max(1, particle_count // 16), max(2, 3 * particle_count // 16)), replace=False).reshape(-1, 4)

        for virtual_site_index, particle_index_1, particle_index_2, particle_index_3 in indices:
            template.setVirtualSite(virtual_site_index, openmm.OutOfPlaneSite(
                particle_index_1,
                particle_index_2,
                particle_index_3,
                *rng.uniform(-1, 1, 3),
            ))

    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, numpy.ones_like(template_indices))
    virtual_sites = {particle_index: combined_system.getVirtualSite(particle_index)
        for particle_index in range(combined_system.getNumParticles()) if combined_system.isVirtualSite(particle_index)}
    
    for instance_index, template_index in enumerate(template_indices):
        particle_offset = particle_offsets[instance_index]
        for particle_index in range(template_sizes[template_index]):
            if templates[template_index].isVirtualSite(particle_index):
                combined_virtual_site = virtual_sites.pop(particle_index + particle_offset)
                reference_virtual_site = templates[template_index].getVirtualSite(particle_index)
                assert isinstance(combined_virtual_site, openmm.OutOfPlaneSite)
                assert combined_virtual_site.getParticle(0) == reference_virtual_site.getParticle(0) + particle_offset
                assert combined_virtual_site.getParticle(1) == reference_virtual_site.getParticle(1) + particle_offset
                assert combined_virtual_site.getParticle(2) == reference_virtual_site.getParticle(2) + particle_offset
                assert combined_virtual_site.getWeight12() == reference_virtual_site.getWeight12()
                assert combined_virtual_site.getWeight13() == reference_virtual_site.getWeight13()
                assert combined_virtual_site.getWeightCross() == reference_virtual_site.getWeightCross()
            else:
                assert particle_index + particle_offset not in virtual_sites

    assert not virtual_sites

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
def test_local_coordinates_site(template_data):
    rng = numpy.random.default_rng((0x3dff101f523c9215, help_deterministic_hash(template_data)))

    template_sizes, template_indices = template_data
    templates = tuple(helpers_test.help_make_templates(template_sizes))

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        index_count = rng.integers(max(1, particle_count // 4), max(2, 3 * particle_count // 4))
        indices = rng.permutation(particle_count)
        slice_indices = [0]
        while slice_indices[-1] < index_count:
            slice_indices.append(slice_indices[-1] + rng.integers(3, 7))
        slice_indices.pop()

        for slice_index_1, slice_index_2 in zip(slice_indices[:-1], slice_indices[1:]):
            virtual_site_index, *particle_indices = indices[slice_index_1:slice_index_2]

            template.setVirtualSite(virtual_site_index, openmm.LocalCoordinatesSite(
                particle_indices,
                help_get_uniform_sum(rng, len(particle_indices), 1),
                help_get_zero_sum(rng, len(particle_indices)),
                help_get_zero_sum(rng, len(particle_indices)),
                rng.uniform(-1, 1, 3),
            ))

    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, numpy.ones_like(template_indices))
    virtual_sites = {particle_index: combined_system.getVirtualSite(particle_index)
        for particle_index in range(combined_system.getNumParticles()) if combined_system.isVirtualSite(particle_index)}
    
    for instance_index, template_index in enumerate(template_indices):
        particle_offset = particle_offsets[instance_index]
        for particle_index in range(template_sizes[template_index]):
            if templates[template_index].isVirtualSite(particle_index):
                combined_virtual_site = virtual_sites.pop(particle_index + particle_offset)
                reference_virtual_site = templates[template_index].getVirtualSite(particle_index)
                assert isinstance(combined_virtual_site, openmm.LocalCoordinatesSite)
                assert combined_virtual_site.getNumParticles() == reference_virtual_site.getNumParticles()
                assert all(combined_virtual_site.getParticle(particle_index) == reference_virtual_site.getParticle(particle_index) + particle_offset
                    for particle_index in range(reference_virtual_site.getNumParticles()))
                assert combined_virtual_site.getOriginWeights() == reference_virtual_site.getOriginWeights()
                assert combined_virtual_site.getXWeights() == reference_virtual_site.getXWeights()
                assert combined_virtual_site.getYWeights() == reference_virtual_site.getYWeights()
            else:
                assert particle_index + particle_offset not in virtual_sites

    assert not virtual_sites

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_PBC_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_harmonic_bond_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0xeb51a94a1d59c3d4, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 2:
            continue

        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            uses_pbc, = class_data_list[class_data_index]

            force = openmm.HarmonicBondForce()
            force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for term_index in range(rng.integers(1, particle_count)):
                force.addBond(*rng.choice(particle_count, 2, replace=False), rng.uniform(1, 2), rng.uniform(10, 20))
            
            template.addForce(force)
   
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_PBC_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_harmonic_angle_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0xb7e5b9e3fe84e1f2, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 3:
            continue

        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            uses_pbc, = class_data_list[class_data_index]

            force = openmm.HarmonicAngleForce()
            force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for term_index in range(rng.integers(1, particle_count)):
                force.addAngle(*rng.choice(particle_count, 3, replace=False), rng.uniform(1, 2), rng.uniform(10, 20))
            
            template.addForce(force)
   
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_PBC_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_periodic_torsion_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0x52c9ecbfd22315c8, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 4:
            continue

        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            uses_pbc, = class_data_list[class_data_index]

            force = openmm.PeriodicTorsionForce()
            force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for term_index in range(rng.integers(1, particle_count)):
                force.addTorsion(*rng.choice(particle_count, 4, replace=False), rng.integers(1, 6), rng.uniform(-numpy.pi, numpy.pi), rng.uniform(10, 20))
            
            template.addForce(force)
   
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_PBC_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_rb_torsion_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0x15e62b53d43795d8, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 4:
            continue

        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            uses_pbc, = class_data_list[class_data_index]

            force = openmm.RBTorsionForce()
            force.setUsesPeriodicBoundaryConditions(uses_pbc)

            for term_index in range(rng.integers(1, particle_count)):
                force.addTorsion(*rng.choice(particle_count, 4, replace=False), *rng.uniform(-10, 10, 6))
            
            template.addForce(force)
   
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_CMAP_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
@pytest.mark.parametrize("share_map_data", (False, True))
def test_cmap_torsion_force(template_data, class_data_list, class_subset, share_map_data):
    rng = numpy.random.default_rng((0xc2b1651b66b9d679, help_deterministic_hash((template_data, class_data_list, class_subset, share_map_data))))

    template_sizes, template_indices = template_data
    if share_map_data:
        temperature_scales = numpy.full(len(template_indices), numpy.exp(rng.uniform(-2, 2)))
        shared_map_data = {}
    else:
        temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 8:
            continue

        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            uses_pbc, map_sizes = class_data_list[class_data_index]
            map_count = len(map_sizes)

            force = openmm.CMAPTorsionForce()
            force.setUsesPeriodicBoundaryConditions(uses_pbc)
            for map_size in map_sizes:
                if share_map_data:
                    if map_size not in shared_map_data:
                        shared_map_data[map_size] = rng.uniform(-10, 10, map_size * map_size)
                    map_data = shared_map_data[map_size]
                else:
                    map_data = rng.uniform(-10, 10, map_size * map_size)
                force.addMap(map_size, map_data)

            for term_index in range(rng.integers(1, particle_count)):
                force.addTorsion(rng.integers(map_count), *rng.choice(particle_count, 8, replace=False))
            
            template.addForce(force)
   
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_CUSTOM_EXTERNAL_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_custom_external_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0x1301ef7b1fa777da, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 1:
            continue

        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            energy_function, global_parameters, particle_parameter_names = class_data_list[class_data_index]

            force = openmm.CustomExternalForce(energy_function)
            for global_parameter_name, global_parameter_value in global_parameters:
                force.addGlobalParameter(global_parameter_name, global_parameter_value)
            for particle_parameter_name in particle_parameter_names:
                force.addPerParticleParameter(particle_parameter_name)

            for term_index in range(rng.integers(1, max(2, particle_count))):
                force.addParticle(rng.integers(particle_count), rng.uniform(-10, 10, len(particle_parameter_names)))
            
            template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(-2, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_CUSTOM_BOND_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_custom_bond_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0xa9952d163503c1d4, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 2:
            continue
        
        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            energy_function, uses_pbc, global_parameters, bond_parameter_names = class_data_list[class_data_index]

            force = openmm.CustomBondForce(energy_function)
            force.setUsesPeriodicBoundaryConditions(uses_pbc)
            for global_parameter_name, global_parameter_value, compute_derivative in global_parameters:
                force.addGlobalParameter(global_parameter_name, global_parameter_value)
                if compute_derivative:
                    force.addEnergyParameterDerivative(global_parameter_name)
            for bond_parameter_name in bond_parameter_names:
                force.addPerBondParameter(bond_parameter_name)

            for term_index in range(rng.integers(1, particle_count)):
                force.addBond(*rng.choice(particle_count, 2, replace=False), rng.uniform(-10, 10, len(bond_parameter_names)))
            
            template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_CUSTOM_ANGLE_TORSION_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_custom_angle_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0x948d89a476f8cb96, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 3:
            continue
        
        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            energy_function, uses_pbc, global_parameters, angle_parameter_names = class_data_list[class_data_index]

            force = openmm.CustomAngleForce(energy_function)
            force.setUsesPeriodicBoundaryConditions(uses_pbc)
            for global_parameter_name, global_parameter_value, compute_derivative in global_parameters:
                force.addGlobalParameter(global_parameter_name, global_parameter_value)
                if compute_derivative:
                    force.addEnergyParameterDerivative(global_parameter_name)
            for angle_parameter_name in angle_parameter_names:
                force.addPerAngleParameter(angle_parameter_name)

            for term_index in range(rng.integers(1, particle_count)):
                force.addAngle(*rng.choice(particle_count, 3, replace=False), rng.uniform(-10, 10, len(angle_parameter_names)))
            
            template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_CUSTOM_ANGLE_TORSION_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_custom_torsion_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0xcf7c2ae1674b5e5a, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 4:
            continue
        
        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            energy_function, uses_pbc, global_parameters, torsion_parameter_names = class_data_list[class_data_index]

            force = openmm.CustomTorsionForce(energy_function)
            force.setUsesPeriodicBoundaryConditions(uses_pbc)
            for global_parameter_name, global_parameter_value, compute_derivative in global_parameters:
                force.addGlobalParameter(global_parameter_name, global_parameter_value)
                if compute_derivative:
                    force.addEnergyParameterDerivative(global_parameter_name)
            for torsion_parameter_name in torsion_parameter_names:
                force.addPerTorsionParameter(torsion_parameter_name)

            for term_index in range(rng.integers(1, particle_count)):
                force.addTorsion(*rng.choice(particle_count, 4, replace=False), rng.uniform(-10, 10, len(torsion_parameter_names)))
            
            template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_CUSTOM_COMPOUND_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_custom_compound_bond_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0x55c72cb9cee9051b, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        
        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            term_particle_count, energy_function, uses_pbc, global_parameters, bond_parameter_names, tabulated_functions = class_data_list[class_data_index]
            if particle_count < term_particle_count:
                continue

            force = openmm.CustomCompoundBondForce(term_particle_count, energy_function)
            force.setUsesPeriodicBoundaryConditions(uses_pbc)
            for global_parameter_name, global_parameter_value, compute_derivative in global_parameters:
                force.addGlobalParameter(global_parameter_name, global_parameter_value)
                if compute_derivative:
                    force.addEnergyParameterDerivative(global_parameter_name)
            for bond_parameter_name in bond_parameter_names:
                force.addPerBondParameter(bond_parameter_name)
            for tabulated_function_name, tabulated_function_type, tabulated_function_args in tabulated_functions:
                force.addTabulatedFunction(tabulated_function_name, tabulated_function_type(*tabulated_function_args))

            for term_index in range(rng.integers(1, particle_count)):
                force.addBond(rng.choice(particle_count, term_particle_count, replace=False), rng.uniform(-10, 10, len(bond_parameter_names)))
            
            template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_CUSTOM_COMPOUND_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_custom_centroid_bond_force(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0x602824bce1cdf3bf, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 2:
            continue
        
        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            term_group_count, energy_function, uses_pbc, global_parameters, bond_parameter_names, tabulated_functions = class_data_list[class_data_index]

            groups = sorted(set(tuple(sorted(rng.choice(particle_count, min(particle_count, rng.integers(2, 6)), replace=False))) for group_index in range(rng.integers(1, particle_count))))
            group_count = len(groups)
            if group_count < term_group_count:
                continue
            groups = [tuple(rng.permutation(groups[group_index])) for group_index in rng.permutation(group_count)]

            for term_group_index in reversed(range(term_group_count)):
                energy_function = energy_function.replace(f"p{term_group_index + 1}", f"g{term_group_index + 1}")
            force = openmm.CustomCentroidBondForce(term_group_count, energy_function)
            force.setUsesPeriodicBoundaryConditions(uses_pbc)
            for global_parameter_name, global_parameter_value, compute_derivative in global_parameters:
                force.addGlobalParameter(global_parameter_name, global_parameter_value)
                if compute_derivative:
                    force.addEnergyParameterDerivative(global_parameter_name)
            for bond_parameter_name in bond_parameter_names:
                force.addPerBondParameter(bond_parameter_name)
            for tabulated_function_name, tabulated_function_type, tabulated_function_args in tabulated_functions:
                force.addTabulatedFunction(tabulated_function_name, tabulated_function_type(*tabulated_function_args))

            for group in groups:
                force.addGroup(group, help_get_uniform_sum(rng, len(group), 1))
            for term_index in range(rng.integers(1, particle_count)):
                force.addBond(rng.choice(group_count, term_group_count, replace=False), rng.uniform(-10, 10, len(bond_parameter_names)))
            
            template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_CUSTOM_NONBONDED_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
@pytest.mark.parametrize("with_groups", (False, True))
def test_custom_nonbonded_force(template_data, class_data_list, class_subset, with_groups):
    rng = numpy.random.default_rng((0x2d9631bbf3c06006, help_deterministic_hash((template_data, class_data_list, class_subset, with_groups))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        template.setDefaultPeriodicBoxVectors(*(7 * numpy.eye(3)))
        particle_count = template.getNumParticles()
        if particle_count < 2:
            continue
        
        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            energy_function, cut_method, uses_switch, uses_long, r_cut, r_switch, compute_parameters, global_parameters, particle_parameter_names, tabulated_functions = class_data_list[class_data_index]

            force = openmm.CustomNonbondedForce(energy_function)
            force.setNonbondedMethod(cut_method)
            force.setUseSwitchingFunction(uses_switch)
            force.setUseLongRangeCorrection(uses_long)
            force.setCutoffDistance(r_cut)
            force.setSwitchingDistance(r_switch)
            for compute_parameter_name, compute_parameter_function in compute_parameters:
                force.addComputedValue(compute_parameter_name, compute_parameter_function)
            for global_parameter_name, global_parameter_value, compute_derivative in global_parameters:
                force.addGlobalParameter(global_parameter_name, global_parameter_value)
                if compute_derivative:
                    force.addEnergyParameterDerivative(global_parameter_name)
            for particle_parameter_name in particle_parameter_names:
                force.addPerParticleParameter(particle_parameter_name)
            for tabulated_function_name, tabulated_function_type, tabulated_function_args in tabulated_functions:
                force.addTabulatedFunction(tabulated_function_name, tabulated_function_type(*tabulated_function_args))

            for particle_index in range(particle_count):
                force.addParticle(rng.uniform(0.5, 1.5, len(particle_parameter_names)))
            
            if with_groups and rng.uniform() < 0.5:
                for group_index in range(rng.integers(2, 5)):
                    group_indices_1 = rng.choice(particle_count, rng.integers(particle_count // 4, 3 * particle_count // 4), replace=False)
                    group_indices_2 = rng.choice(particle_count, rng.integers(particle_count // 4, 3 * particle_count // 4), replace=False)
                    force.addInteractionGroup(group_indices_1, group_indices_2)
            
            exclusions = set()
            exclusion_count = min(rng.integers(particle_count, 2 * particle_count), particle_count * (particle_count - 1) // 4)
            while len(exclusions) < exclusion_count:
                exclusions.add(tuple(sorted(rng.choice(particle_count, 2, replace=False))))
            exclusions = sorted(exclusions)
            for exclusion_index in rng.permutation(exclusion_count):
                force.addExclusion(*rng.permutation(exclusions[exclusion_index]))

            template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 7, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_data", HELP_TEMPLATE_SIZES_INDICES_PARTICLES)
@pytest.mark.parametrize("class_data_list", HELP_FORCE_GROUP_CLASS_DATA_LIST)
@pytest.mark.parametrize("class_subset", (False, True))
def test_force_groups(template_data, class_data_list, class_subset):
    rng = numpy.random.default_rng((0xeb51a94a1d59c3d4, help_deterministic_hash((template_data, class_data_list, class_subset))))

    template_sizes, template_indices = template_data
    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    templates = tuple(helpers_test.help_make_templates(template_sizes))
    
    class_data_count = len(class_data_list)

    for template_index, template in enumerate(templates):
        particle_count = template.getNumParticles()
        if particle_count < 2:
            continue

        for class_data_index in rng.choice(class_data_count, rng.integers(max(1, class_data_count // 2), max(2, class_data_count)) if class_subset else class_data_count, replace=False):
            force_group = class_data_list[class_data_index]

            force = openmm.HarmonicBondForce()
            force.setForceGroup(force_group)

            for term_index in range(rng.integers(1, particle_count)):
                force.addBond(*rng.choice(particle_count, 2, replace=False), rng.uniform(1, 2), rng.uniform(10, 20))
            
            template.addForce(force)
   
    combined_system, particle_offsets = multiopenmm.stacking.stack(templates, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)))
    for force_group in range(32):
        help_check_combined_context(temperature_scales, templates, template_indices, combined_system, particle_offsets, rng.uniform(0, 2, (particle_offsets[-1], 3)), force_group=force_group)

def test_discrete_1d_function():
    rng = numpy.random.default_rng((0xbc4a45c0c3de43dd,))

    template, = helpers_test.help_make_templates((6,))
    force = openmm.CustomCompoundBondForce(2, "table(i)/distance(p1,p2)")
    force.addTabulatedFunction("table", openmm.Discrete1DFunction(numpy.sin(numpy.arange(3))))
    force.addPerBondParameter("i")
    for i in range(3):
        force.addBond((2 * i, 2 * i + 1), (i,))
    template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack((template,), (0,), (1,))
    help_check_combined_context((1,), (template,), (0,), combined_system, particle_offsets, rng.uniform(-1, 1, (particle_offsets[-1], 3)))

def test_discrete_2d_function():
    rng = numpy.random.default_rng((0xf232145d8053a489,))

    template, = helpers_test.help_make_templates((30,))
    force = openmm.CustomCompoundBondForce(2, "table(i,j)/distance(p1,p2)")
    force.addTabulatedFunction("table", openmm.Discrete2DFunction(3, 5, numpy.sin(numpy.arange(15))))
    force.addPerBondParameter("i")
    force.addPerBondParameter("j")
    for i in range(3):
        for j in range(5):
            k = i * 5 + j
            force.addBond((2 * k, 2 * k + 1), (i, j))
    template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack((template,), (0,), (1,))
    help_check_combined_context((1,), (template,), (0,), combined_system, particle_offsets, rng.uniform(-1, 1, (particle_offsets[-1], 3)))

def test_discrete_3d_function():
    rng = numpy.random.default_rng((0x9b4ba1c05c8b5fa,))

    template, = helpers_test.help_make_templates((210,))
    force = openmm.CustomCompoundBondForce(2, "table(i,j,k)/distance(p1,p2)")
    force.addTabulatedFunction("table", openmm.Discrete3DFunction(3, 5, 7, numpy.sin(numpy.arange(105))))
    force.addPerBondParameter("i")
    force.addPerBondParameter("j")
    force.addPerBondParameter("k")
    for i in range(3):
        for j in range(5):
            for k in range(7):
                l = (i * 5 + j) * 7 + k
                force.addBond((2 * l, 2 * l + 1), (i, j, k))
    template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack((template,), (0,), (1,))
    help_check_combined_context((1,), (template,), (0,), combined_system, particle_offsets, rng.uniform(-1, 1, (particle_offsets[-1], 3)))

def test_continuous_1d_function():
    rng = numpy.random.default_rng((0x5d9574403aca1fc5,))

    template, = helpers_test.help_make_templates((200,))
    force = openmm.CustomCompoundBondForce(2, "table(distance(p1,p2))")
    force.addTabulatedFunction("table", openmm.Continuous1DFunction(numpy.sin(numpy.arange(10)), 0, 4, False))
    for i in range(100):
        force.addBond((2 * i, 2 * i + 1))
    template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack((template,), (0,), (1,))
    help_check_combined_context((1,), (template,), (0,), combined_system, particle_offsets, rng.uniform(-1, 1, (particle_offsets[-1], 3)))

def test_continuous_2d_function():
    rng = numpy.random.default_rng((0xaf8aa9fc8cf4f6a8,))

    template, = helpers_test.help_make_templates((300,))
    force = openmm.CustomCompoundBondForce(3, "table(distance(p1,p2),distance(p2,p3))")
    force.addTabulatedFunction("table", openmm.Continuous2DFunction(10, 10, numpy.sin(numpy.arange(100)), 0, 4, 0, 4, False))
    for i in range(100):
        force.addBond((3 * i, 3 * i + 1, 3 * i + 2))
    template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack((template,), (0,), (1,))
    help_check_combined_context((1,), (template,), (0,), combined_system, particle_offsets, rng.uniform(-1, 1, (particle_offsets[-1], 3)))

def test_continuous_3d_function():
    rng = numpy.random.default_rng((0x2c4c447c633e4c50,))

    template, = helpers_test.help_make_templates((300,))
    force = openmm.CustomCompoundBondForce(3, "table(distance(p1,p2),distance(p2,p3),distance(p1,p3))")
    force.addTabulatedFunction("table", openmm.Continuous3DFunction(10, 10, 10, numpy.sin(numpy.arange(1000)), 0, 4, 0, 4, 0, 4, False))
    for i in range(100):
        force.addBond((3 * i, 3 * i + 1, 3 * i + 2))
    template.addForce(force)
    
    combined_system, particle_offsets = multiopenmm.stacking.stack((template,), (0,), (1,))
    help_check_combined_context((1,), (template,), (0,), combined_system, particle_offsets, rng.uniform(-1, 1, (particle_offsets[-1], 3)))

@pytest.mark.parametrize("template_indices", HELP_TEMPLATE_INDICES_SAMPLE)
def test_stack_forces(template_indices):
    rng = numpy.random.default_rng((0xf06bb5a84b266ced, help_deterministic_hash((template_indices,))))

    temperature_scales = numpy.exp(rng.uniform(-2, 2, len(template_indices)))
    combined_system, particle_offsets = multiopenmm.stacking.stack(HELP_SAMPLE_SYSTEMS, template_indices, temperature_scales)
    help_check_combined_context(temperature_scales, HELP_SAMPLE_SYSTEMS, template_indices, combined_system, particle_offsets,
        numpy.concatenate([HELP_SAMPLE_POSITIONS[template_index] for template_index in template_indices]) if template_indices else numpy.zeros((0, 3)))
