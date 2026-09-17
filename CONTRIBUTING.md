# Contributing to TorchDR

Thank you for helping improve TorchDR. Contributions may include bug reports,
feature proposals, documentation, examples, tests, performance investigations,
code review, and implementation work.

All participants must follow the [Code of Conduct](CODE_OF_CONDUCT.md). Project
roles and decision making are described in [GOVERNANCE.md](GOVERNANCE.md).

## Before opening a pull request

- Search the [issue tracker](https://github.com/TorchDR/TorchDR/issues) for
  related work.
- Open an issue before investing in a substantial feature, public API change,
  or architectural change so the approach can be discussed early.
- For a focused bug fix, documentation correction, or small maintenance change,
  a pull request may be opened directly.

## Development setup

Fork the repository, create a dedicated branch, and install the development and
documentation dependencies:

```bash
git clone git@github.com:YOUR-USERNAME/TorchDR.git
cd TorchDR
python -m pip install -e ".[dev,doc]"
pre-commit install
```

## Making a change

- Follow the existing style and public API conventions.
- Add or update tests when the change affects stable user-visible behavior or a
  reusable correctness invariant.
- Update user-facing documentation when behavior or APIs change.
- Keep pull requests focused and explain the motivation and resulting behavior.
- Use a `[Type] Description` pull request title, where `Type` is `Feature`,
  `Fix`, `Refactor`, `Doc`, `Test`, or `Chore`.

Run the relevant tests while developing. Before requesting review, run:

```bash
pre-commit run --all-files
python -m pytest torchdr/tests
```

Optional backends and distributed behavior have additional CI coverage. Include
the relevant local or cluster validation in the pull request when changing
those paths.

## Review and acceptance

Maintainers evaluate correctness, compatibility, scope, tests, documentation,
and long-term maintenance cost. At least one maintainer other than the author
should approve a change before merge, when practical. Larger or user-visible
changes may require broader discussion under the governance decision process.

The rendered contribution guide contains additional details and contributor
tips: https://torchdr.github.io/dev/torchdr.contributing.html
