import jax
import jax.random as jrandom
import jax.numpy as jnp

def generate_ar_timeseries(key, N, T, phi=0.8, sigma=0.7):
    """
    Generate N AR(1) time series of length T, output shape (T, N)
    """
    key_noise = jrandom.split(key, N)
    
    def single_ar(noise_key):
        innovations = jrandom.normal(noise_key, (T,)) * sigma
        series = jnp.zeros(T)
        
        def step(state, t):
            return phi * state + innovations[t], phi * state + innovations[t]
        
        _, series = jax.lax.scan(step, 0., jnp.arange(T))
        return series
    
    # Generate N series in parallel and reshape
    series = jax.vmap(single_ar)(key_noise)
    return series.T  # Now returns shape (T, N)

def generate_normal_timeseries(key, N, T):
    """
    Generate N time series from a normal distribution of length T.
    Output shape: (T, N)
    
    Args:
        key: JAX random key
        N (int): Number of time series
        T (int): Length of each time series
        mean (float): Mean of the normal distribution
        std (float): Standard deviation of the normal distribution
        
    Returns:
        jnp.ndarray: Shape (T, N) containing the generated time series
    """
    return jrandom.normal(key, shape=(T, N))

def generate_poisson_timeseries(key, N, T, rate=1.0):
    """
    Generate N Poisson process time series of length T, output shape (T, N)
    """
    return jrandom.poisson(key, shape=(T, N), lam=rate)

def generate_exp_growth_timeseries(key, N, T, growth_rate=0.1, noise_scale=0.1):
    """
    Generate N exponential growth time series of length T, output shape (T, N)
    """
    time = jnp.arange(T)
    base_curve = jnp.exp(growth_rate * time)
    noise = jnp.exp(jrandom.normal(key, (T, N)) * noise_scale)
    return base_curve[:, None] * noise

def generate_seasonal_timeseries(key, N, T, period=24, amplitude=1.0, trend=0.0, noise_scale=0.1):
    """
    Generate N seasonal time series of length T, output shape (T, N)
    """
    time = jnp.arange(T)
    seasonal = amplitude * jnp.sin(2 * jnp.pi * time / period)
    trend_component = trend * time
    base_curve = seasonal + trend_component
    noise = jrandom.normal(key, (T, N)) * noise_scale
    return base_curve[:, None] + noise