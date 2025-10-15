# tests/test_main_cli.py
from click.testing import CliRunner
from walker.main import cli

def test_index_command(test_env):
    """
    Test the main 'index' command to ensure it runs and finds files.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ['index'])

    assert result.exit_code == 0
    assert "Starting scan" in result.output
    assert "Found 3 files to process" in result.output
    assert "Indexing complete" in result.output

    # Now test a report to see if the data is there
    result = runner.invoke(cli, ['type-summary'])
    assert result.exit_code == 0
    assert "text/plain" in result.output
    assert "3" in result.output # 3 files of type text/plain

def test_find_dupes_command(test_env):
    """
    Test the 'find-dupes' command to ensure it correctly identifies duplicates.
    """
    runner = CliRunner()
    # First, run the indexer
    runner.invoke(cli, ['index'])

    # Now, find duplicates
    result = runner.invoke(cli, ['find-dupes'])

    assert result.exit_code == 0
    assert "Found 1 set of duplicate files" in result.output
    assert "file1.txt" in result.output
    assert "file3_dupe.txt" in result.output