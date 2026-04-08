import jax
import jax.numpy as jnp
import flax.nnx as nnx
import logging
from .PatchTST import PatchTST
from .models import SinusodialPosEmb, TCN, RNNBlock, BiRNN, RNN
from typing import Optional
from copy import deepcopy
logging.basicConfig(level=logging.INFO)

class Flow(nnx.Module):
    """
    Sequence-to-sequence flow model with time conditioning and a latent recurrent core.

    This variant:
      - encodes past context from (optionally) observations + controls using a temporal encoder
        (BiRNN or TCN depending on `modeltype`)
      - conditions a latent sequence model (`hidden_model`, a BiRNN/LSTM) on:
          * latent input `z` (augmented with repeated context embedding `h`)
          * future control inputs for the prediction segment
          * a sinusoidal time embedding
      - decodes the latent sequence back to `dim_y` with a 1x1 convolution.

    High-level data flow
    --------------------
    1) Project observation history and controls to `dim_hidden` via 1x1 convolutions.
    2) Encode the past (context window) into a single context vector `h`.
    3) Repeat `h` over the prediction horizon (`segment_length_steps`).
    4) Build per-step inputs for the latent model: [latent(z,h), future_controls, time_embedding].
    5) Run the latent sequence model and decode to predictions.

    Notes
    -----
    - This class assumes time is the leading dimension for sequences (T, features).
    - `dropout_rngs` is forwarded to recurrent components that apply dropout.
    """

    def __init__(self, runconfig, rngs: nnx.Rngs):
        """
        Parameters
        ----------
        runconfig : Any
            Configuration object expected to define at least:
              - dim_y : int, observation dimension
              - dim_u : int, control/stimulus dimension
              - dim_t : int, time embedding dimension
              - dim_hidden : int, hidden/channel dimension
              - latent_hidden : int, latent sequence feature dimension
              - hidden_layers : int, number of encoder layers/blocks
              - latent_blocks : int, number of latent recurrent layers/blocks
              - kernel_size : int, for TCN (if modeltype == "tcn")
              - dropout : float, dropout rate (used in recurrent/TCN modules)
              - modeltype : str, e.g. "tcn", "lstm", "gru", "simple"
              - both : bool, whether to condition on observations + controls (True)
                        or controls only (False)
              - context_length_steps : int, number of timesteps in the context window
              - segment_length_steps : int, number of timesteps to predict

        rngs : nnx.Rngs
            RNG container used for initializing NNX modules (and dropout collections).
        """
        self.dim_y, self.dim_u, self.dim_t = runconfig.dim_y, runconfig.dim_u, runconfig.dim_t
        self.dim_hidden = runconfig.dim_hidden
        self.modeltype = runconfig.modeltype
        self.context_length_steps, self.segment_length_steps = runconfig.context_length_steps, runconfig.segment_length_steps
        self.time_emb = SinusodialPosEmb(self.dim_t)
        self.runconfig = runconfig
        self.both = runconfig.both
        self._init_encoder_layers(rngs)
        self._init_encoder_and_latent_model(rngs)
        self._init_decoder(rngs)
        
    def _init_encoder_layers(self, rngs):
        rc = self.runconfig
        self.input_layer = nnx.Conv(self.dim_y, rc.dim_hidden, kernel_size=1, rngs=rngs)        
        self.control_layer = nnx.Conv(self.dim_u, rc.dim_hidden, kernel_size=1, rngs=rngs)
        
    def _init_encoder_and_latent_model(self, rngs):
        rc = self.runconfig
        if rc.modeltype == "tcn":
            self.linear_in = nnx.Linear(self.dim_y +self.dim_hidden, rc.latent_hidden, rngs=rngs)
            self.encoder = TCN(
                num_inputs=rc.dim_hidden * 2,
                num_channels=[rc.dim_hidden] * rc.hidden_layers,
                kernel_size=rc.kernel_size,
                dropout=rc.dropout,
                rngs=rngs
            )
        
        else: 
            self.linear_in = nnx.Linear(self.dim_y +self.dim_hidden * 2, rc.latent_hidden, rngs=rngs)
            if rc.both: 
                self.encoder = BiRNN(rc.dim_hidden, rc.dim_hidden, 0, rc.dim_hidden,  rngs=rngs, modeltype=rc.modeltype, dropout= rc.dropout, num_layers = rc.hidden_layers )
            else: 
                self.encoder = BiRNN(rc.dim_hidden//2, rc.dim_hidden, 0, rc.dim_hidden//2,  rngs=rngs, modeltype=rc.modeltype, dropout= rc.dropout, num_layers = rc.hidden_layers )
        self.hidden_model = BiRNN(rc.latent_hidden, rc.latent_hidden//2, self.dim_t, rc.dim_hidden,  rngs=rngs, modeltype="lstm", dropout= rc.dropout, num_layers = rc.latent_blocks )
        
    def _init_decoder(self, rngs):
        rc = self.runconfig
        self.output_layer = nnx.Conv(rc.latent_hidden, self.dim_y, kernel_size=1, rngs=rngs)

    def encode_past(self, x_history: Optional[jnp.ndarray], c_history: jnp.ndarray, dropout_rngs):
        """
        Encode the past context window into a single context vector.
        """
        if self.both:

            x = jnp.concatenate([x_history, c_history], axis=-1) 
            
        else: 
            x = c_history
        if self.modeltype == "tcn":
            x = self.encoder(x)
        else:             
            x = self.encoder(x, dropout_rngs)
        
        return x[-1]

    @nnx.jit
    def __call__(self, z, x_history, c, t, dropout_rngs):
        x_history = self.input_layer(x_history)
        c = self.control_layer(c)    

        h = self.encode_past(x_history, c[:self.context_length_steps], dropout_rngs)
        h = jnp.repeat(h[None, :], self.segment_length_steps, axis=0)
        time_emb = self.time_emb(t)
        time_emb = jnp.repeat(time_emb[None, :], h.shape[0], axis=0)
        z = nnx.gelu(self.linear_in(jnp.concatenate([z, h], axis = -1)))
        x = jnp.concatenate([z, c[self.context_length_steps:], time_emb], axis=1) 
       
        x = self.hidden_model(x, dropout_rngs)
                 
        x = self.output_layer(x)
        return x