from jax import random
from lantern_npyro.model import model
from numpyro.infer import (
    MCMC,
    NUTS,
)

def test_5_samples(sim_data):
    _, X, _, Y, _, _ = sim_data
    rng_key  = random.PRNGKey(0)
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=2,
        num_samples=5,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
