"""Tests for TorchDR CLI module."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
from unittest.mock import patch, MagicMock

from torchdr.cli import get_gpu_count, main


class TestGetGpuCount:
    """Tests for get_gpu_count function."""

    def test_returns_device_count(self):
        """Test that GPU count is returned correctly."""
        # get_gpu_count uses torch.cuda.device_count internally
        count = get_gpu_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_function_handles_errors(self):
        """Test that function doesn't crash on edge cases."""
        # Just verify function returns an int without crashing
        result = get_gpu_count()
        assert isinstance(result, int)


class TestMain:
    """Tests for main CLI function."""

    @patch("torchdr.cli.subprocess.run")
    @patch("torchdr.cli.get_gpu_count")
    def test_all_gpus(self, mock_gpu_count, mock_run):
        """Test running with all available GPUs."""
        mock_gpu_count.return_value = 4
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", ["torchdr", "script.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "torchrun" in cmd
        assert "--nproc_per_node=4" in cmd
        assert "script.py" in cmd

    @patch("torchdr.cli.subprocess.run")
    @patch("torchdr.cli.get_gpu_count")
    def test_specific_gpu_count(self, mock_gpu_count, mock_run):
        """Test running with specific number of GPUs."""
        mock_gpu_count.return_value = 8
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", ["torchdr", "--gpus", "2", "script.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        cmd = mock_run.call_args[0][0]
        assert "--nproc_per_node=2" in cmd

    @patch("torchdr.cli.get_gpu_count")
    def test_no_gpus_error(self, mock_gpu_count):
        """Test error when no GPUs available with 'all' flag."""
        mock_gpu_count.return_value = 0

        with patch("sys.argv", ["torchdr", "script.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    @patch("torchdr.cli.get_gpu_count")
    def test_invalid_gpus_argument(self, mock_gpu_count):
        """Test error on invalid --gpus value."""
        mock_gpu_count.return_value = 4

        with patch("sys.argv", ["torchdr", "--gpus", "invalid", "script.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    @patch("torchdr.cli.subprocess.run")
    @patch("torchdr.cli.get_gpu_count")
    def test_gpus_exceeds_available(self, mock_gpu_count, mock_run, capsys):
        """Test warning when requesting more GPUs than available."""
        mock_gpu_count.return_value = 2
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", ["torchdr", "--gpus", "8", "script.py"]):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        assert "Warning" in captured.err
        # Should use only 2 GPUs
        cmd = mock_run.call_args[0][0]
        assert "--nproc_per_node=2" in cmd

    @patch("torchdr.cli.subprocess.run")
    @patch("torchdr.cli.get_gpu_count")
    def test_script_args_passed(self, mock_gpu_count, mock_run):
        """Test that script arguments are passed through."""
        mock_gpu_count.return_value = 1
        mock_run.return_value = MagicMock(returncode=0)

        with patch(
            "sys.argv",
            ["torchdr", "script.py", "--data", "/path", "--epochs", "10"],
        ):
            with pytest.raises(SystemExit):
                main()

        cmd = mock_run.call_args[0][0]
        assert "--data" in cmd
        assert "/path" in cmd
        assert "--epochs" in cmd
        assert "10" in cmd

    @patch("torchdr.cli.subprocess.run")
    @patch("torchdr.cli.get_gpu_count")
    def test_torchrun_not_found(self, mock_gpu_count, mock_run):
        """Test error when torchrun is not found."""
        mock_gpu_count.return_value = 1
        mock_run.side_effect = FileNotFoundError()

        with patch("sys.argv", ["torchdr", "script.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
