Overview
========

MultiOpenMM is a package that enables setting up and running parallel molecular
dynamics simulations, including but not limited to replica exchange simulations,
using the OpenMM molecular simulation package.  MultiOpenMM supports running
independent simulations of multiple, possibly topologically distinct, molecular
systems, within a single OpenMM context, as well as flexibly distributing
parallel simulation workloads across multiple clients whose availabilities may
vary over the course of a simulation.

.. toctree::
   api
   cli
   guide
