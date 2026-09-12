import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session")
def qmodel():
    from golden.quant import QConfig, build
    return build(QConfig())
