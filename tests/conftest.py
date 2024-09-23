import pytest
from lantern_npyro.data import sim


@pytest.fixture
def sim_data():
    return sim(10)
