How to Contribute
=================


``TorchDR`` is a community-driven project and welcomes contributions of all forms.

We will acknowledge all contributors in the documentation and in the source code. 
Significant contributions will also be taken into account when deciding 
on the authorship of future publications.

The preferred way to contribute to ``TorchDR`` is to fork the `main
repository <https://github.com/torchdr/torchdr/>`_ on GitHub,
then submit a "Pull Request" (PR). When preparing the PR, please make sure to
check the following points:

- The code is compliant with the `black <https://github.com/psf/black>`_ style. This can be done easily by installing the black library and running ``black .`` in the root directory of the repository after making the desired changes.
- The automatic tests pass on your local machine. This can be done by running ``python -m pytest torchdr/tests``in the root directory of the repository after making the desired changes.
- The documentation is updated if necessary. After making the desired changes, this can be done in the directory ``docs`` by running one of the commands in the table below.

.. list-table::
   :widths: 40 50
   :header-rows: 1

   * - Command
     - Description of command
   * - ``make html``
     - Generates all the documentation
   * - ``make html-noplot``
     - Generates documentation faster but without plots
   * - ``PATTERN=/path/to/file make html-pattern``
     - Generates documentation for files matching ``$(PATTERN)``

If you are not familiar with the GitHub contribution workflow, you can also open 
an issue on the `issue tracker <https://github.com/torchdr/torchdr/issues>`_. 
We will then try to address the issue as soon as possible.