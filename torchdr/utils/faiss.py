"""Robust handling of faiss as optional dependency."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import sys
import warnings


def _check_faiss_installation():
    """
    Check if FAISS is properly installed and provide helpful error messages.

    Returns
    -------
    faiss : module or None
        The faiss module if properly installed, None otherwise.
    error_msg : str or None
        Error message if FAISS is not properly installed, None otherwise.
    """
    try:
        import faiss as _faiss_module

        # Check 1: Detect wrong package (faiss==1.12.0 from pip imports as bool)
        if not isinstance(_faiss_module, type(sys)):
            return None, (
                "FAISS imported incorrectly (wrong type detected). "
                "This typically means you installed the broken 'faiss' package from pip."
            )

        # Check 2: Verify essential functions exist
        if not hasattr(_faiss_module, "IndexFlatL2"):
            return None, (
                "FAISS module is incomplete (missing IndexFlatL2). "
                "This indicates a broken or partial installation."
            )

        # Check 3: Try to create a basic index to verify it works
        try:
            test_index = _faiss_module.IndexFlatL2(10)
            del test_index  # Clean up
        except Exception as e:
            return None, f"FAISS index creation failed: {str(e)}"

        # All checks passed
        return _faiss_module, None

    except ImportError:
        return None, "FAISS is not installed."
    except Exception as e:
        return None, f"Unexpected error importing FAISS: {str(e)}"


# Perform the check
faiss, _error_message = _check_faiss_installation()

# If FAISS is not available or broken, show a helpful warning
if faiss is None and _error_message:
    _install_instructions = """
Please install FAISS via conda (recommended):

  For GPU support:
    conda install -c pytorch -c nvidia faiss-gpu=1.11.0

  For CPU only:
    conda install -c pytorch faiss-cpu=1.11.0

IMPORTANT: Do NOT use 'pip install faiss' - the package on PyPI (version 1.12.0)
is broken and incomplete. Always install via conda from the pytorch channel.

See: https://github.com/TorchDR/TorchDR#installation
"""

    warnings.warn(
        f"\n{'=' * 78}\n"
        f"FAISS Error: {_error_message}\n"
        f"{_install_instructions}"
        f"{'=' * 78}\n",
        ImportWarning,
        stacklevel=2,
    )

    # Set faiss to False for backward compatibility with existing code
    faiss = False
