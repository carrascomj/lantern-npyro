import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def kernel(X, Z, var=0.1):
    X_scaled = X
    Z_scaled = Z
    X_norms = jnp.sum(X_scaled ** 2, axis=1)
    Z_norms = jnp.sum(Z_scaled ** 2, axis=1)
    deltaXsq = X_norms[:, None] + Z_norms[None, :] - 2 * (X_scaled @ Z_scaled.T)
    k = var * jnp.exp(-0.5 * deltaXsq)
    return k


def model(X, Y):
    p = X.shape[-1]
    latent = 8
    with numpyro.plate("latent", size=latent):
        # precision hyperprior, hierarchical over latent dimension
        alpha = numpyro.sample("alpha", dist.Gamma(0.001, 0.001))
        W = numpyro.sample("weight_matrix", dist.Normal(jnp.zeros(p), 1 / alpha[:, None]).to_event(1))

    z = numpyro.deterministic("z", X @ W.T)
    # compute kernel
    k = kernel(z, z)
    noise_var = numpyro.sample("noise_var", dist.HalfCauchy(scale=1.0))
    cov = k + noise_var

    # sample Y according to the standard gaussian process formula
    f = numpyro.sample(
        "f",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=cov),
    )
    noise_unk = numpyro.sample("noise_unk", dist.HalfCauchy(scale=1.0))
    numpyro.sample("y", dist.Normal(f, noise_unk), obs=Y)
