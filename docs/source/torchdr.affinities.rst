.. _affinities:

Affinities
=============

Affinities are :class:`torch.nn.Module` that take a noisy image as input and return a denoised image.
They can be used as a building block for plug-and-play restoration, for building unrolled architectures,
or as a standalone denoiser. All denoisers have a ``forward`` method that takes a noisy image and a noise level
(which generally corresponds to the standard deviation of the noise) as input and returns a denoised image:

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