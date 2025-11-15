"""TorchDR command-line interface for easy multi-GPU execution."""

import argparse
import subprocess
import sys


def get_gpu_count():
    """Get the number of available GPUs.

    Returns
    -------
    int
        Number of GPUs available on the system.
    """
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def run_command():
    """Run a Python script with multi-GPU support using torchrun.

    This command wraps PyTorch's torchrun to provide a user-friendly interface
    for launching single-node multi-GPU training jobs.

    Examples
    --------
    Run with 4 GPUs:
        torchdr run --gpus 4 my_script.py --arg1 value1

    Use all available GPUs:
        torchdr run --gpus all my_script.py
    """
    parser = argparse.ArgumentParser(
        description="Run a Python script with TorchDR multi-GPU support (single-node)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all available GPUs (default)
  torchdr run my_script.py

  # Use 4 GPUs
  torchdr run --gpus 4 my_script.py

  # Pass arguments to your script (after the script name)
  torchdr run --gpus 2 my_script.py --data-path /path/to/data

Note: Your script will automatically use distributed mode when launched with
this command. No manual setup needed - just import and use TorchDR normally!
Arguments after the script name are passed directly to your script.
        """,
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help="Number of GPUs to use (default: all available GPUs)",
    )

    parser.add_argument("script", help="Python script to run")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the script",
    )

    args = parser.parse_args()

    # Detect available GPUs
    available_gpus = get_gpu_count()

    # Determine number of GPUs to use
    if args.gpus.lower() == "all":
        n_gpus = available_gpus
        if n_gpus == 0:
            print("Error: No GPUs detected on this system", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            n_gpus = int(args.gpus)
        except ValueError:
            print(
                f"Error: --gpus must be a number or 'all', got '{args.gpus}'",
                file=sys.stderr,
            )
            sys.exit(1)

        if n_gpus > available_gpus:
            print(
                f"Warning: Requested {n_gpus} GPUs but only {available_gpus} "
                f"available. Using {available_gpus}.",
                file=sys.stderr,
            )
            n_gpus = available_gpus

    # Print hardware detection summary
    print("TorchDR Multi-GPU Launcher")
    print(f"  Hardware detected: {available_gpus} GPU(s)")
    print(f"  GPUs to use: {n_gpus}")
    print("  Backend: nccl")

    # Build torchrun command (single-node only)
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        args.script,
    ]
    cmd.extend(args.script_args)

    # Print command for debugging
    print(f"Executing: {' '.join(cmd)}")
    print()

    # Execute torchrun
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(
            "Error: torchrun not found. Make sure PyTorch is installed.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


def main():
    """Main entry point for the TorchDR CLI."""
    if len(sys.argv) < 2:
        print("Usage: torchdr <command> [options]", file=sys.stderr)
        print("\nAvailable commands:", file=sys.stderr)
        print("  run    Run a script with multi-GPU support", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        # Remove 'run' from sys.argv so argparse sees the right arguments
        sys.argv.pop(1)
        run_command()
    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print("\nAvailable commands:", file=sys.stderr)
        print("  run    Run a script with multi-GPU support", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
