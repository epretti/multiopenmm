Python API
==========

Simulation setup
----------------

.. autoclass:: multiopenmm.Simulation
   :members:

.. autoclass:: multiopenmm.Ensemble
   :members:

.. autoclass:: multiopenmm.CanonicalEnsemble
   :members:
   :show-inheritance:

.. autoclass:: multiopenmm.IsothermalIsobaricEnsemble
   :members:
   :show-inheritance:

.. autoclass:: multiopenmm.Precision()
   :members:
   :member-order: bysource

.. autoclass:: multiopenmm.Thermostat()
   :members:
   :member-order: bysource

.. autoclass:: multiopenmm.Barostat()
   :members:
   :member-order: bysource

Running and restarting
----------------------

Task management
---------------

Stacking customization
----------------------

.. autofunction:: multiopenmm.stack.stack

.. autoclass:: multiopenmm.stack.VirtualSiteProcessor
   :members:

.. autodata:: multiopenmm.stack.DefaultVirtualSiteProcessor
   :annotation:

.. autoclass:: multiopenmm.stack.VirtualSiteHandler
   :members:

.. autoclass:: multiopenmm.stack.ForceProcessor
   :members:

.. autoclass:: multiopenmm.stack.ForceHandler
   :members:

.. autodata:: multiopenmm.stack.DefaultForceProcessor
   :annotation:
