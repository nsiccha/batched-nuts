"""The usual way to run many NUTS chains in JAX: `jax.vmap` over `blackjax.nuts`."""

import blackjax
import jax
import jax.numpy as jnp


def make_blackjax_sampler(logdensity, max_depth=10):
    """Returns `sample(jobs, num_draws, data=None)`, which runs all chains in step.

    Chain `j` makes the transitions `blackjax.nuts(...).step(fold_in(key[j], t), state)`
    for `t = 0, 1, ...`, the same transitions as `batched_nuts.sample`.  Under `vmap`
    every transition lasts as long as the longest trajectory among the chains.
    """

    def chain(transitions, data, key, position, step_size, inverse_mass, dataset):
        if data is None:
            target = logdensity
        else:
            data_slice = jax.tree.map(lambda x: x[dataset], data)
            target = lambda position: logdensity(position, data_slice)
        nuts = blackjax.nuts(target, step_size, inverse_mass, max_num_doublings=max_depth)

        def transition(state, t):
            state, info = nuts.step(jax.random.fold_in(key, t), state)
            return state, dict(
                position=state.position,
                logdensity=state.logdensity,
                num_integration_steps=info.num_integration_steps,
                is_divergent=info.is_divergent,
                is_turning=info.is_turning,
                acceptance_rate=info.acceptance_rate,
            )

        return jax.lax.scan(transition, nuts.init(position), transitions)[1]

    @jax.jit
    def run(jobs, transitions, data):
        dataset = jnp.zeros(len(jobs.position), jnp.int32) if data is None else jobs.dataset
        return jax.vmap(lambda *job: chain(transitions, data, *job))(
            jobs.key, jobs.position, jobs.step_size, jobs.inverse_mass, dataset
        )

    def sample(jobs, num_draws, data=None):
        return run(jobs, jnp.arange(num_draws), data)

    return sample
