.. _affinities:

Affinities
=============

All affinities inherit the structure of the :meth:`Affinity` class:

.. currentmodule:: torchdr.affinity

.. autoclass:: Affinity
   :members:
   :undoc-members:
   :show-inheritance:


They can be used as a building block for plug-and-play restoration, for building unrolled architectures


   >>> import torch

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.GibbsAffinity
   torchdr.affinity.StudentAffinity

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.EntropicAffinity
   torchdr.affinity.SymmetricEntropicAffinity
   torchdr.affinity.L2SymmetricEntropicAffinity
   torchdr.affinity.DoublyStochasticEntropic