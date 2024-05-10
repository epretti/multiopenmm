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

Default values
^^^^^^^^^^^^^^

.. autodata:: multiopenmm.parallel.DEFAULT_VECTOR_LENGTH
.. autodata:: multiopenmm.parallel.DEFAULT_MINIMIZE_TOLERANCE
.. autodata:: multiopenmm.parallel.DEFAULT_MINIMIZE_ITERATION_COUNT
.. autodata:: multiopenmm.parallel.DEFAULT_STEP_LENGTH
.. autodata:: multiopenmm.parallel.DEFAULT_CONSTRAINT_TOLERANCE
.. autodata:: multiopenmm.parallel.DEFAULT_TEMPERATURE
.. autodata:: multiopenmm.parallel.DEFAULT_PRESSURE
.. autodata:: multiopenmm.parallel.DEFAULT_THERMOSTAT_STEPS
.. autodata:: multiopenmm.parallel.DEFAULT_BAROSTAT_STEPS

Running and restarting
----------------------

Manual trajectory exporting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: multiopenmm.IntegrationResult
   :members:

.. autofunction:: multiopenmm.export_results

.. autofunction:: multiopenmm.delete_results

.. autoclass:: multiopenmm.export.Exporter
   :members:

.. autoclass:: multiopenmm.DCDExporter
   :members:
   :show-inheritance:

.. autoclass:: multiopenmm.TextEnergyExporter
   :members:
   :show-inheritance:

Replica exchange run management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: multiopenmm.parallel.ExchangePairGenerator
   :members:

.. autoclass:: multiopenmm.RandomAdjacentExchangePairGenerator
   :members:
   :show-inheritance:

.. autoclass:: multiopenmm.parallel.AcceptanceCriterion
   :members:

.. autoclass:: multiopenmm.MetropolisAcceptanceCriterion
   :members:
   :show-inheritance:

Automatic restart management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Task management
---------------

.. autoclass:: multiopenmm.Manager
   :members:

.. autoclass:: multiopenmm.SynchronousManager
   :members:
   :show-inheritance:

.. autoclass:: multiopenmm.PlatformData
   :members:

Stacking customization
----------------------

.. autofunction:: multiopenmm.stacking.stack

Virtual site handling
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: multiopenmm.stacking.VirtualSiteProcessor
   :members:

.. autodata:: multiopenmm.stacking.DefaultVirtualSiteProcessor
   :annotation:

.. autoclass:: multiopenmm.stacking.VirtualSiteHandler
   :members:

Force handling
^^^^^^^^^^^^^^

.. autoclass:: multiopenmm.stacking.ForceProcessor
   :members:

.. autodata:: multiopenmm.stacking.DefaultForceProcessor
   :annotation:

.. autoclass:: multiopenmm.stacking.ForceHandler
   :members:

Tabulated function handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: multiopenmm.stacking.TabulatedFunctionProcessor
   :members:

.. autodata:: multiopenmm.stacking.DefaultTabulatedFunctionProcessor
   :annotation:

.. autoclass:: multiopenmm.stacking.TabulatedFunctionHandler
   :members:

Miscellaneous
-------------

.. autoclass:: multiopenmm.MultiOpenMMError
   :members:

.. autoclass:: multiopenmm.MultiOpenMMWarning
   :members:

.. autofunction:: multiopenmm.get_scratch_directory

.. autofunction:: multiopenmm.set_scratch_directory
