from typing import Optional
import jax.numpy as jnp
import flax.nnx as nnx
import jax 
from .layers import TimeConditionedDense
from .models import TCNEncoder, BiRNN, RNN
import logging
logging.basicConfig(level=logging.INFO)

class Flow(nnx.Module):
    """
    Flow model for predicting next fMRI activity (or latent state)
    conditioned on past observations and stimulus/control inputs.
    """

    def __init__(self, runconfig, rngs: nnx.Rngs):
        """
        Parameters
        ----------
        runconfig : Any
            Run/config object holding model hyperparameters and dataset/model settings.

        rngs : nnx.Rngs
            RNG container used by Flax NNX

        Notes
        -----
        Initialization is split into helpers:
          - `_init_encoders`: subject-specific encoders for stimulus and optionally observations
          - `_init_model_core`: temporal encoder (BiRNN/TCN) based on `runconfig.modeltype`
          - `_init_latent_blocks`: stack of time-conditioned blocks operating on latent `z`
          - `_init_decoder`: subject-specific output heads
          - `_init_norms_and_dropout`: layer norms and dropout layers (if used by the encoder)
        """
        super().__init__()
        self.dim_y, self.dim_u, self.dim_t = runconfig.dim_y, runconfig.dim_u, runconfig.dim_t
        self.hidden_layers, self.dim_hidden = runconfig.hidden_layers, runconfig.dim_hidden
        self.runconfig = runconfig
        self.modeltype = runconfig.modeltype
        self.both = runconfig.both
        self.rngs = rngs
        self._init_encoders(rngs)
        self._init_model_core(rngs)
        self._init_latent_blocks(rngs)
        self._init_decoder(rngs)
        self._init_norms_and_dropout(rngs)
        
        
    def _make_linear(self, in_dim: int, out_dim: int, n_subj, rngs):
            return [nnx.Linear(in_dim, out_dim, rngs=rngs) for _ in range(n_subj)]

    def _init_encoders(self, rngs):
        rc = self.runconfig
        n_subj = (
            max(rc.training_subject_ids) + 1 if rc.foundation == 4 else 1
        )   
        if rc.both:  # observation + stimulus
            self.observation_encoder = self._make_linear(
                rc.dim_y, rc.dim_hidden // 2, n_subj, rngs
            )
            self.input_encoder = self._make_linear(
                rc.dim_u, rc.dim_hidden // 2, n_subj, rngs
            )
        else:  # stimulus only
            logging.info("Only using stimulus as context")
            self.input_encoder = self._make_linear(
                rc.dim_u, rc.dim_hidden, n_subj, rngs
            )
        self.linear_in = nnx.Linear(self.dim_y, rc.latent_hidden, rngs=rngs)
        
    def _init_model_core(self, rngs):
        rc = self.runconfig
        cell_map = {
            "lstm": nnx.nn.recurrent.OptimizedLSTMCell,
            "gru": nnx.nn.recurrent.GRUCell,
            "simple": nnx.nn.recurrent.SimpleCell,
        }

        if rc.modeltype in cell_map:
            self.rnn_encoder = BiRNN(rc.dim_hidden //2, rc.dim_hidden //2, 0, rc.dim_hidden//2,  rngs=rngs, modeltype=rc.modeltype, dropout= rc.dropout, num_layers = rc.hidden_layers )
            # self.rnn_encoder = RNN(rc.dim_hidden //2, rc.dim_hidden, 0, control_dim=rc.dim_hidden //2,  rngs=rngs, modeltype=rc.modeltype, dropout= rc.dropout, num_layers = rc.hidden_layers, use_layernorm= True )

        elif rc.modeltype == "tcn":
            self.tcn_encoder = TCNEncoder(
                dim_in=rc.dim_hidden,
                dim_hidden=rc.dim_hidden,
                kernel_size=rc.kernel_size,
                n_blocks=rc.hidden_layers,
                dropout = rc.dropout,
                rngs=rngs
                )
        else:
            raise ValueError(f"Modeltype '{rc.modeltype}' not implemented")

    def _init_latent_blocks(self, rngs):
        rc = self.runconfig
        self.blocks = [TimeConditionedDense(rc.latent_hidden, rc.dim_hidden, rc.dim_t, rngs) 
                      for _ in range(rc.latent_blocks)]
    def _init_decoder(self, rngs):
        rc = self.runconfig
        n_subj = (
            max(rc.training_subject_ids) + 1 if (rc.foundation == 3 or rc.foundation == 4) else 1
        )
        self.linear_out = self._make_linear(
            rc.latent_hidden + rc.dim_hidden, rc.dim_y, n_subj, rngs
        )
    def _init_norms_and_dropout(self, rngs):
        rc = self.runconfig
        self.layernorms = [nnx.LayerNorm(rc.dim_hidden, rngs=rngs) for _ in range(rc.hidden_layers)]

        self.dropout_layers = [nnx.Dropout(rate=rc.dropout, rng_collection='recurrent_dropout', rngs=rngs) for _ in range(rc.hidden_layers)]

    def apply_encoder(self, sid, encoders, c):
        """
        Apply a subject-specific encoder to an input vector
        """
        return nnx.gelu(jax.lax.switch((sid), encoders, c))
    
            
    def encode_past(self, x_history: Optional[jnp.ndarray], c: jnp.ndarray, subject_id, dropout_rngs = None) -> jnp.ndarray:
        """
        Encodes past observations and control inputs into a single context vector
        """
        sid = subject_id if len(self.input_encoder) > 1 else 0 
        encoded_inputs = nnx.gelu(jax.lax.switch(sid, self.input_encoder, c))
    
        if self.both is True: 
            encoded_obs = nnx.gelu(jax.lax.switch(sid, self.observation_encoder, x_history))
            encoded_inputs = jnp.concatenate([encoded_obs, encoded_inputs], axis=-1) 

        if self.modeltype in ("lstm", "gru", "simple"):
            outputs = self.rnn_encoder(encoded_inputs, dropout_rngs)
            
            return outputs[-1]
    
        elif self.modeltype == "tcn":
            return self.tcn_encoder(encoded_inputs)

    @nnx.jit
    def __call__(self, z, x_history, c, t, subject_id = None, dropout_rngs = None):
        h = self.encode_past(x_history, c, subject_id, dropout_rngs)
        z = nnx.gelu(self.linear_in(z))
        for block in self.blocks:
            z = block(z, h, t)
            z = nnx.gelu(z) + z  
        
        sid = subject_id
        concat_input_final = jnp.concatenate([z, h], axis=-1)
        x_next = jax.lax.switch(sid, self.linear_out, concat_input_final)

        return x_next
    