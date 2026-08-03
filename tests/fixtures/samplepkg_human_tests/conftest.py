"""Make the fixture package importable wherever this tests directory lands.

The directory above this one (tests/fixtures locally, the workspace root once
copied into a mutation or coverage workspace) contains ``samplepkg``; putting it
on sys.path mirrors how a real library's tests sit next to its package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
