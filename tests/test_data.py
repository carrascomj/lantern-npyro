from lantern_npyro.data import sim_2d

def test_sim_has_expected_shapes(sim_data):
    W, X, z, y, Z, f = sim_data
    assert W.shape[0] == X.shape[1]
    # data surface shapes
    assert X.shape[0] == z.shape[0] == y.shape[0]
    # interpolation result shapes
    assert f.shape[0] == Z.shape[0]


def test_sim2d_makes_sense():
    W, X, z, y, Z, f = sim_2d(222)
    assert X.shape[0] == z.shape[0] == y.shape[0]
