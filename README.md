MultiOpenMM: Flexible parallel molecular dynamics with OpenMM
=============================================================

🚧 🚧 **This library is a work in progress; some features may be incomplete,
incompletely documented, or incompletely tested.  This file will be updated as
development progresses.** 🚧 🚧 

MultiOpenMM is a package that enables setting up and running parallel molecular
dynamics simulations, including but not limited to replica exchange simulations,
using the OpenMM molecular simulation package.  MultiOpenMM supports running
independent simulations of multiple, possibly topologically distinct, molecular
systems, within a single OpenMM context.

This can be especially useful when running, *e.g.*, replica exchange simulations
with **many** replicas but **few** particles per replica: ordinary use of OpenMM
with one replica per GPU, for instance, would require many available GPUs but
would waste most of their computational capacity.  In this case, depending on
the interactions present in the system of interest, MultiOpenMM can be used to
saturate a single GPU by stacking many non-interacting replicas into a single
OpenMM `System`.  Scaling of interactions to account for differences in
temperature is handled automatically by MultiOpenMM.  The package even supports
more flexible use cases than replica exchange, allowing systems stacked within a
single OpenMM `System` to have different force field terms and parameters or
molecular topologies (as might be useful for some kinds of high-throughput
screening simulations across chemical or protein sequence space).  MultiOpenMM
attempts to optimize the resulting set of OpenMM `Force` objects in the stack to
maximize performance.

Alternatively, in cases where each replica is large, or otherwise cannot be
stacked (MultiOpenMM cannot stack systems with long-range electrostatics or
barostats, for instance), it may be desired to distribute work across multiple
GPUs.  MultiOpenMM contains a client-server system that allows simulation tasks
for any number of replicas or systems to be distributed across any (possibly
non-commensurate) number of OpenMM workers.  Workers can even be added and
removed dynamically during the course of a simulation, and simulation tasks for
workers that fail will be automatically queued for redistribution to other
workers.

Requirements
------------

[Python](https://www.python.org/) (at least 3.10), [OpenMM](https://openmm.org/)
8, [NumPy](https://numpy.org/), [SciPy](https://scipy.org).
[MDTraj](https://www.mdtraj.org) is required for DCD trajectory export.

Documentation
-------------

Sphinx API documentation is available and can be compiled from `/doc/` with,
*e.g.*, `make html`.

Acknowledgment and license information
--------------------------------------

This software is developed by Evan Pretti in the [Shell research
group](https://theshelllab.org/) in the [Department of Chemical
Engineering](https://www.chemengr.ucsb.edu/) at the [University of California,
Santa Barbara](https://www.ucsb.edu/).  Evan Pretti gratefully acknowledges
support from the Molecular Sciences Software Institute (MolSSI) as well as
helpful discussions with Levi Naden at MolSSI.  The Molecular Sciences Software
Institute is supported by the National Science Foundation under Grant
CHE-2136142.

This software is ©2024 The Regents of the University of California.  All rights
reserved.  This software is licensed under the MIT License.  For license
information, see the [license file](LICENSE.md).
