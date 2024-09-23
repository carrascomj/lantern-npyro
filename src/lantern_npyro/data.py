from itertools import combinations
import numpy as np
from jax.random import PRNGKey, randint
import jax.numpy as jnp

import numpyro.distributions as dist


def sigmoid(x, scale=100):
    return 1 / (1 + jnp.exp(-x*scale))

# squared exponential kernel with diagonal noise term
def test_kernel(X, Z):
    D = X.shape[-1]
    # Ensure X and Z are both of shape [N, D], where D = 1 in your case
    # Reshape if necessary
    X = X.reshape(-1, D)  # Shape: [N, 1]
    Z = Z.reshape(-1, D)  # Shape: [N, 1]
    
    # Compute pairwise differences
    deltaX = (X - Z.T)  # Shape: [N, N]
    
    # Compute squared distances
    deltaXsq = deltaX ** 2  # Shape: [N, N]
    
    # Compute the kernel matrix
    k = 5 * jnp.exp(-0.5 * deltaXsq)   
    return k


def test_kernel_2(X, Z, var=0.1):
    X_scaled = X
    Z_scaled = Z
    X_norms = jnp.sum(X_scaled ** 2, axis=1)
    Z_norms = jnp.sum(Z_scaled ** 2, axis=1)
    deltaXsq = X_norms[:, None] + Z_norms[None, :] - 2 * (X_scaled @ Z_scaled.T)
    k = var * jnp.exp(-0.5 * deltaXsq)
    return k

def sim(seed, p=5):
    """Simulate data for testing the model.
    
    Taken from LANTERN: https://lantern-gpl.readthedocs.io/en/latest/examples/example-1d.html
    """

    N = 2 ** p
    rng = PRNGKey(seed)

    W = (dist.Normal(jnp.zeros((p, 1)), 1.0).rsample(rng) - 0.2 * 2.0)

    X = np.zeros((N, p))
    ind = 1

    # for all # of mutations
    for mutations in range(1, p + 1):

        # for selected combination of mutations for a variant
        for variant in combinations(range(p), mutations):

            # for each selected
            for s in variant:
                X[ind, s] = 1

            # update after variant
            ind += 1

    # z is the mutations for our data
    z = X @ W
    # Z is random points in the surface that we interpolate
    Z = np.linspace(z.min(), z.max(), 100)[:, None]
    z_samp = jnp.concat((z, Z), axis=0)
    K = test_kernel(z_samp, z_samp)

    sigma_sq = 0.0025
    kern = sigma_sq * K
    diag = jnp.eye(N + 100) * 1e-7
    cov = kern + diag


    f = dist.MultivariateNormal(
        loc=jnp.zeros(N + 100), covariance_matrix=cov
    ).rsample(rng)# + sigmoid(0.9 + z_samp[:, 0])

    y = f[:N] + dist.Normal(jnp.zeros(N), 1.0).rsample(rng) * 0.05

    return W, X, z, y, Z, f[N:]


def sim_2d(seed, p=5):
    """Simulate data for testing the model.
    
    Taken from LANTERN: https://lantern-gpl.readthedocs.io/en/latest/examples/example-1d.html
    """

    N = 2 ** p
    latent = p
    rng = PRNGKey(seed)

    W = (dist.Normal(jnp.zeros((p, latent)), 1.0).rsample(rng) - 0.2 * 2.0)

    X = np.zeros((N, p))
    ind = 1

    # for all # of mutations
    for mutations in range(1, p + 1):

        # for selected combination of mutations for a variant
        for variant in combinations(range(p), mutations):

            # for each selected
            for s in variant:
                X[ind, s] = 1

            # update after variant
            ind += 1

    # z is the mutations for our data
    z = X @ W
    # Z is random points in the surface that we interpolate
    Z = jnp.linspace(z.min(), z.max(), 100 * latent).reshape(100, latent)
    z_samp = jnp.concat((z, Z), axis=0)
    K = test_kernel_2(z_samp, z_samp)

    diag = jnp.eye(N + 100) * 1e-4
    cov = K + diag


    from jax.scipy.stats import multivariate_normal as mvn
    f = dist.MultivariateNormal(
        loc=jnp.zeros(N + 100), covariance_matrix=cov
    ).rsample(rng) + sigmoid(0.9 + z_samp[:, 0])

    y = f[:N] + dist.Normal(jnp.zeros(N), 1.0).rsample(rng) * 0.05

    return W, X, z, y, Z, f[N:]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    p = 5

    W, X, z, y, Z, f = sim_2d(100, p=p)

    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(Z, f)
    plt.scatter(z[:, 0], y, c="C2", alpha=0.8)
    plt.scatter(z[:, 1], y, c="C3", alpha=0.8)
    plt.scatter(z[:, 2], y, c="C4", alpha=0.8)
    plt.scatter(z[:, 3], y, c="C5", alpha=0.8)
    plt.scatter(z[:, 4], y, c="C6", alpha=0.8)
    plt.axvline(0, c="k", ls="--")

    for i in range(p):
        plt.arrow(0, -.05*i, W[i,0].item(), 0, color=f"C{3+i}", width=0.01)

    plt.ylabel("phenotype")
    plt.xlabel("$z_1$")
    plt.show()
