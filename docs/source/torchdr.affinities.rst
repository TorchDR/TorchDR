.. _affinities:

Affinities
=============

Base structure and simple examples
-----------------------------------

All affinities inherit the structure of the :meth:`Affinity` class.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.affinity.Affinity

If computations can be performed in log domain, the :meth:`LogAffinity` class should be used.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.affinity.LogAffinity


They can be used as a building block for DR algorithms.

   >>> import torch

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.GibbsAffinity
   torchdr.affinity.StudentAffinity


Affinities based on entropic normalization
------------------------------------------


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.EntropicAffinity
   torchdr.affinity.SymmetricEntropicAffinity
   torchdr.affinity.L2SymmetricEntropicAffinity
   torchdr.affinity.DoublyStochasticEntropic


Other various affinities
-------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.DoublyStochasticQuadratic