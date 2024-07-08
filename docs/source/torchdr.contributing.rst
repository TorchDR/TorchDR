How to Contribute
=================

Thank you for considering contributing to ``TorchDR``.
This library is a community-driven project and welcomes contributions of all forms.

We will acknowledge all contributors in the documentation and in the source code. 
Significant contributions will also be taken into account when deciding 
on the authorship of future publications.

The preferred way to contribute to ``TorchDR`` is to fork the `main
repository <https://github.com/torchdr/torchdr/>`_ on GitHub,
then submit a "Pull Request" (PR). When preparing the PR, please make sure to
check the following points:

- The code is compliant with the `black <https://github.com/psf/black>`_ style. This can be done easily by installing the black library and running ``black .`` in the root directory of the repository after making the desired changes.
- All public methods should have informative docstrings.
- The automatic tests pass on your local machine. This can be done by running ``python -m pytest torchdr/tests`` in the root directory of the repository after making the desired changes.
- If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.
- The documentation is updated if necessary. You can edit the documentation using any text editor and then generate the HTML output by typing ``make html`` from the ``docs/`` directory. Alternatively, make can be used to quickly generate the documentation without the example gallery with ``make html-noplot``. The resulting HTML files will be placed in ``docs/build/html/`` and are viewable in a web browser.

If you are not familiar with the GitHub contribution workflow, you can also open 
an issue on the `issue tracker <https://github.com/torchdr/torchdr/issues>`_. 
We will then try to address the issue as soon as possible.


New contributor tips
--------------------

A great way to start contributing to ``TorchDR`` is to pick an item
from the list of `good first issues <https://github.com/TorchDR/TorchDR/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_ in the issue tracker. Resolving these issues allow you to start
contributing to the project without much prior knowledge.