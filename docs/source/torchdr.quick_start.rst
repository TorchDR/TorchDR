.. _quick_start:

.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:


Quick Start Guide
=================


Installation
------------
To install TorchDR, run in the terminal:

.. code-block:: shell

   pip install torchdr


When to use TorchDR
-----------------------

- For leveraging the power of **GPU acceleration** for faster computations. All the modules in TorchDR are designed to work seamlessly on the GPU by setting ``device = 'cuda'``.
- For comparing different dimensionality reduction methods in a fair and reproducible way. TorchDR maximizes **code sharing** across various methods, ensuring a **fair benchmarking** that emphasizes core differences.
- For developing new dimensionality reduction approaches. TorchDR provides a **modular and extensible framework** that allows you to focus on the core ideas of your method, while the rest of the pipeline is taken care of.


.. minigallery:: torchdr.AffinityMatcher
    :add-heading: Examples using ``AffinityMatcher``:
