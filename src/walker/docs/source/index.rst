Walker File Indexer
===================

.. include:: ../../../../README.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Command-Line Interface
----------------------

The following documents all the available commands and their options.

.. click:: walker.main:cli
   :prog: poetry run python -m walker.main
   :nested: full

API Reference
-------------

This section provides detailed documentation for the core modules of the application, generated directly from the source code docstrings.

.. automodule:: walker.file_processor
   :members:

.. automodule:: walker.reporter
   :members:

.. automodule:: walker.indexer
   :members: