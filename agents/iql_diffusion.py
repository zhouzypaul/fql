import copy
import math
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import MLP, Value


def target_update(model, target_model, tau):
    """Interpolate from model to target_model. Tau = ratio of current model to target model"""
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wraps a function to supply jax rng. It will remember the rng state for that function.
    """
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapped


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def timestep_embedding(t, emb_size=16, max_period=10000):
    """t is between [0, 1]."""
    t = jax.lax.convert_element_type(t, jnp.float32)
    t = t * max_period
    dim = emb_size
    half = dim // 2
    freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embedding


class DiffusionPolicy(nn.Module):
    """Input is a noised action, observation, and t."""
    hidden_dims: Any
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)

    @nn.compact
    def __call__(self, obs, is_positive, noised_action, t):
        t_embedding = timestep_embedding(t)
        is_positive_embedding = nn.Embed(2, 32)(is_positive)
        concat_input = jnp.concatenate([obs, noised_action, t_embedding, is_positive_embedding], axis=-1)
        outputs = MLP(self.hidden_dims, activate_final=True, **self.mlp_kwargs)(concat_input)
        v = nn.Dense(self.action_dim)(outputs)
        return v


class IQLDiffusionAgent(flax.struct.PyTreeNode):
    """IQL agent with diffusion policy and CFG."""
    rng: Any
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def critic_loss(self, batch, critic_params=None):
        """Compute the critic loss."""
        next_v = self.value(batch['next_observations'])
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        qs = self.critic(batch['observations'], batch['actions'], params=critic_params)
        critic_loss = ((qs - target_q[None])**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs[0].mean(),
        }

    def value_loss(self, batch, value_params=None):
        """Compute the value loss."""
        qs = self.target_critic(batch['observations'], batch['actions'])
        q = jnp.min(qs, axis=0)  # Min over ensemble.
        v = self.value(batch['observations'], params=value_params)
        value_loss = expectile_loss(q-v, self.config['expectile']).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'v_min': v.min(),
            'v_max': v.max(),
        }

    def actor_loss(self, batch, actor_params=None, rng=None):
        """Compute the actor loss."""
        if rng is None:
            rng = self.rng
        eps_rng, time_rng = jax.random.split(rng, 2)
        
        actions = batch['actions']
        # Ensure observations_policy is available
        batch_with_policy = dict(batch)
        if 'observations_policy' not in batch_with_policy:
            batch_with_policy['observations_policy'] = batch['observations']

        v = self.value(batch_with_policy['observations_policy'])
        if self.config['target_extraction']:
            qs = self.target_critic(batch_with_policy['observations_policy'], actions)
        else:
            qs = self.critic(batch_with_policy['observations_policy'], actions)
        q = jnp.min(qs, axis=0)  # Min over ensemble.
        exp_a = jnp.exp((q - v) * self.config['temperature'])
        exp_a = jnp.minimum(exp_a, 100.0)
        exp_a = ((q-v) > 0).astype(jnp.float32)

        x1 = actions
        x0 = jax.random.normal(eps_rng, x1.shape)
        t = jax.random.randint(time_rng, (x1.shape[0],), 0, self.config['denoise_steps'] + 1).astype(jnp.float32) / self.config['denoise_steps']
        tv = t[..., None]
        x_t = x0 * (1 - tv) + x1 * tv
        vel = (x1 - x0)

        # Positive samples
        idx_positive = jnp.ones((x1.shape[0],), dtype=jnp.int32)
        pred_vel_positive = self.actor(batch_with_policy['observations_policy'], idx_positive, x_t, t, params=actor_params)
        vel_loss_positive = jnp.mean(((vel - pred_vel_positive)**2), axis=-1) * exp_a

        # Unconditional samples
        idx_uncond = jnp.zeros((x1.shape[0],), dtype=jnp.int32)
        pred_vel_uncond = self.actor(batch_with_policy['observations_policy'], idx_uncond, x_t, t, params=actor_params)
        vel_loss_uncond = jnp.mean(((vel - pred_vel_uncond)**2), axis=-1)

        actor_loss = jnp.mean(vel_loss_positive + vel_loss_uncond * 0.1)

        return actor_loss, {
            'actor_q': q.mean(),
            'actor_loss': actor_loss,
            'actor_positive_ratio': ((q-v) > 0).mean(),
            'actor_loss_positive': vel_loss_positive.mean(),
            'actor_loss_uncond': vel_loss_uncond.mean(),
            'actor_losses': vel_loss_positive + vel_loss_uncond * 0.1,
        }

    @jax.jit
    def update(agent, batch):
        new_rng, actor_rng = jax.random.split(agent.rng, 2)

        def critic_loss_fn(critic_params):
            return agent.critic_loss(batch, critic_params)
        
        def value_loss_fn(value_params):
            return agent.value_loss(batch, value_params)
        
        def actor_loss_fn(actor_params):
            return agent.actor_loss(batch, actor_params, rng=actor_rng)
        
        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn)
        new_target_critic = target_update(agent.critic, agent.target_critic, agent.config['tau'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn)
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn)

        return agent.replace(rng=new_rng, critic=new_critic, target_critic=new_target_critic, value=new_value, actor=new_actor), {
            **critic_info, **value_info, **actor_info
        }

    @jax.jit
    def sample_actions(agent, observations: np.ndarray, *, seed: Any, temperature: float = 1., cfg = 1.) -> jnp.ndarray:
        observations = observations[None]

        x = jax.random.normal(seed, (observations.shape[0], agent.config['action_dim']))
        dt = 1.0 / agent.config['denoise_steps']
        idx_positive = jnp.ones((x.shape[0],), dtype=jnp.int32)
        idx_uncond = jnp.zeros((x.shape[0],), dtype=jnp.int32)
        for t in range(agent.config['denoise_steps']):
            ti = jnp.ones((x.shape[0],)) * (t / agent.config['denoise_steps'])
            v_positive = agent.actor(observations, idx_positive, x, ti)
            v_uncond = agent.actor(observations, idx_uncond, x, ti)
            v = v_uncond + cfg * (v_positive - v_uncond)
            x = x + v*dt
        actions = x[0]
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss for compatibility with main.py evaluation."""
        if rng is None:
            rng = self.rng
        
        critic_loss, critic_info = self.critic_loss(batch)
        value_loss, value_info = self.value_loss(batch)
        actor_loss, actor_info = self.actor_loss(batch, rng=rng)

        total_loss = critic_loss + value_loss + actor_loss
        info = {**critic_info, **value_info, **actor_info}
        
        return total_loss, info

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config,
    ):
        """Create a new IQL diffusion agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = ex_actions.shape[-1]
        config = dict(config)  # Make a mutable copy
        config['action_dim'] = action_dim
        
        # Get activation function
        try:
            activation_fn = getattr(nn, config['activation'])
        except:
            activation_fn = nn.gelu
        print(f"Using activation function: {activation_fn}")

        actor_def = DiffusionPolicy(config['actor_hidden_dims'], action_dim, mlp_kwargs=dict(activations=activation_fn, layer_norm=config['use_layer_norm']))
        
        if config['opt_decay_schedule'] == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-config['actor_lr'], config['max_steps'])
            actor_tx = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=config['actor_lr'])

        idx_positive = jnp.ones((ex_actions.shape[0],), dtype=jnp.int32)
        actor_params = actor_def.init(actor_key, ex_observations, idx_positive, ex_actions, jnp.zeros(ex_actions.shape[0],))['params']
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

        critic_def = Value(config['hidden_dims'], layer_norm=config['use_layer_norm'], num_ensembles=config['num_qs'])
        critic_params = critic_def.init(critic_key, ex_observations, ex_actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=config['critic_lr']))
        target_critic = TrainState.create(critic_def, critic_params)

        value_def = Value(config['hidden_dims'], layer_norm=config['use_layer_norm'], num_ensembles=1)
        value_params = value_def.init(value_key, ex_observations)['params']
        value = TrainState.create(value_def, value_params, tx=optax.adam(learning_rate=config['value_lr']))

        config_dict = flax.core.FrozenDict(**config)
        return cls(rng, critic=critic, target_critic=target_critic, value=value, actor=actor, config=config_dict)


def get_config():
    """Get default configuration for IQL diffusion agent."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='iql_diffusion',
            batch_size=256,
            goal_conditioned=0,
            actor_lr=3e-4,
            value_lr=3e-4,
            critic_lr=3e-4, 
            num_qs=2,
            actor_hidden_dims=(512, 512, 512, 512),
            hidden_dims=(512, 512, 512, 512),
            discount=0.99,
            expectile=0.9,
            temperature=3.0,  # 0 for behavior cloning.
            dropout_rate=0,
            use_tanh=0,
            state_dependent_std=0,
            use_layer_norm=1,
            activation='gelu',
            fixed_std=0,
            tau=0.005,
            opt_decay_schedule='none',
            target_extraction=1,
            denoise_steps=16,
            objective='awr',
            action_distribution='data',
            max_steps=500000,  # Added for cosine schedule
        )
    )
    return config
