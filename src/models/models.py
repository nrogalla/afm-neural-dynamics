# Modeltypes such as RNN, TCN, etc
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from flax.nnx import Dropout
import jax
import flax.nnx as nnx
from typing import List
from jax import lax
import logging
from .layers import SinusodialPosEmb
logging.basicConfig(level=logging.INFO)
 
class RNNBlock(nnx.Module):
    """
    Single bidirectional RNN block (forward + backward) with optional dropout and layer norm.
    """
    def __init__(self, dim_in: int, dim_out: int, time_dim: int, control_dim:int,  rngs: nnx.Rngs, modeltype="simple", dropout = 0):
        """
        Parameters
        ----------
        dim_in : int
            Base input feature dimension (excluding time/control features).

        dim_out : int
            Hidden size of each directional RNN cell. The returned feature dimension
            per timestep is `2 * dim_out` due to concatenation of forward and backward
            outputs.

        time_dim : int
            Size of per-timestep time features concatenated to the input.

        control_dim : int
            Size of per-timestep control/stimulus features concatenated to the input.

        rngs : nnx.Rngs
            RNG container for parameter initialization (and dropout modules).

        modeltype : {"simple", "gru", "lstm"}, default="simple"
            Type of recurrent cell used in both directions.

        dropout : float, default=0
            Dropout rate applied to outputs of each direction.
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.time_dim = time_dim
        self.modeltype = modeltype
        if modeltype == "lstm":
            self.rnn_fwd = nnx.nn.recurrent.OptimizedLSTMCell(dim_in +  time_dim + control_dim, dim_out, rngs=rngs)
            self.rnn_bwd = nnx.nn.recurrent.OptimizedLSTMCell(dim_in +  time_dim + control_dim, dim_out, rngs=rngs)
        elif modeltype == "simple":
            self.rnn_fwd = nnx.nn.recurrent.SimpleCell(dim_in + time_dim + control_dim, dim_out, rngs=rngs)
            self.rnn_bwd = nnx.nn.recurrent.SimpleCell(dim_in + time_dim + control_dim, dim_out, rngs=rngs)
        elif modeltype == "gru":
            self.rnn_fwd = nnx.nn.recurrent.GRUCell(dim_in + time_dim + control_dim, dim_out, rngs=rngs)
            self.rnn_bwd = nnx.nn.recurrent.GRUCell(dim_in + time_dim + control_dim, dim_out, rngs=rngs)

        self.dropout_fwd = Dropout(dropout, rngs=rngs)
        self.dropout_bwd = Dropout(dropout, rngs=rngs)
        self.layer_norm_fwd = nnx.LayerNorm(dim_out, rngs=rngs)
        self.layer_norm_bwd = nnx.LayerNorm(dim_out, rngs=rngs)
        self.rngs = rngs

    def __call__(self, x, dropout_rngs):
        def scan_fn(carry, x_t):
            carry_fw, carry_bw = carry
            carry_fw, out_fw = self.rnn_fwd(carry_fw, x_t)
            out_fw = self.layer_norm_fwd(out_fw)
            carry_bw, out_bw = self.rnn_bwd(carry_bw, x_t[::-1])
            out_bw = self.layer_norm_bwd(out_bw)

            out_fw = nnx.Dropout(self.dropout_fwd.rate)(out_fw, rngs=nnx.Rngs({'dropout': dropout_rngs}))
            out_bw = nnx.Dropout(self.dropout_bwd.rate)(out_bw, rngs=nnx.Rngs({'dropout': dropout_rngs}))

            return (carry_fw, carry_bw), (out_fw, out_bw)

        hidden_dim = self.rnn_fwd.hidden_features
        
        if self.modeltype == "lstm":
            initial_carry = (
                (jnp.zeros((hidden_dim,), x.dtype), jnp.zeros((hidden_dim,), x.dtype)),
                (jnp.zeros((hidden_dim,), x.dtype), jnp.zeros((hidden_dim,), x.dtype)),
            )
        else:
            initial_carry = (
                jnp.zeros((hidden_dim,), x.dtype),
                jnp.zeros((hidden_dim,), x.dtype),
            )

        scan_inputs = (x)#, keys_fwd, keys_bwd)
        _, (outputs_fw, outputs_bw) = jax.lax.scan(scan_fn, initial_carry, scan_inputs)
        outputs = jnp.concatenate([outputs_fw, outputs_bw[::-1]], axis=-1)
        return outputs

class BiRNN(nnx.Module):
    """
    Multi-layer bidirectional RNN encoder.
    """
    def __init__(
        self, dim_in: int, dim_out: int, time_dim: int, control_dim: int,
        rngs: nnx.Rngs, modeltype="simple", dropout=0, num_layers=2, 
        use_layernorm=False,
    ):
        """
        Parameters
        ----------
        dim_in : int
            Base input feature dimension (excluding time/control features).

        dim_out : int
            Hidden size for each cell in each direction. Output feature dimension per
            timestep is `2 * dim_out`.

        time_dim : int
            Size of per-timestep time features concatenated to inputs.

        control_dim : int
            Size of per-timestep control/stimulus features concatenated to inputs.

        rngs : nnx.Rngs
            RNG container for parameter initialization.

        modeltype : {"simple", "gru", "lstm"}, default="simple"
            Recurrent cell type.

        dropout : float, default=0
            Dropout rate applied after each layer’s cell output.

        num_layers : int, default=2
            Number of stacked bidirectional layers.

        use_layernorm : bool, default=False
            If True, apply LayerNorm after each directional cell output.
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.time_dim = time_dim
        self.control_dim = control_dim
        self.modeltype = modeltype
        self.num_layers = num_layers
        self.rngs = rngs

        input_dim = dim_in + time_dim + control_dim

        # Initialize forward and backward RNN cell stacks
        self.rnn_fwd = []
        self.rnn_bwd = []
        self.layer_norm_fwd = []
        self.layer_norm_bwd = []
        self.dropout_fwd = []
        self.dropout_bwd = []

        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else dim_out
            if modeltype == "lstm":
                fwd_cell = nnx.nn.recurrent.OptimizedLSTMCell(layer_input_dim, dim_out, rngs=rngs)
                bwd_cell = nnx.nn.recurrent.OptimizedLSTMCell(layer_input_dim, dim_out, rngs=rngs)
            elif modeltype == "gru":
                fwd_cell = nnx.nn.recurrent.GRUCell(layer_input_dim, dim_out, rngs=rngs)
                bwd_cell = nnx.nn.recurrent.GRUCell(layer_input_dim, dim_out, rngs=rngs)
            else:
                fwd_cell = nnx.nn.recurrent.SimpleCell(layer_input_dim, dim_out, rngs=rngs)
                bwd_cell = nnx.nn.recurrent.SimpleCell(layer_input_dim, dim_out, rngs=rngs)

            self.rnn_fwd.append(fwd_cell)
            self.rnn_bwd.append(bwd_cell)
            if use_layernorm:
                self.layer_norm_fwd.append(nnx.LayerNorm(dim_out, rngs=rngs))
                self.layer_norm_bwd.append(nnx.LayerNorm(dim_out, rngs=rngs))
            else: 
                self.layer_norm_fwd.append(None)
                self.layer_norm_bwd.append(None)
            
            self.dropout_fwd.append(Dropout(dropout, rngs=rngs))
            self.dropout_bwd.append(Dropout(dropout, rngs=rngs))

    def __call__(self, x, dropout_rngs):
        hidden_dim = self.dim_out

        def init_state(cell):
            if self.modeltype == "lstm":
                return (jnp.zeros((hidden_dim,), x.dtype), jnp.zeros((hidden_dim,), x.dtype))
            else:
                return jnp.zeros((hidden_dim,), x.dtype)

        init_carry_fw = [init_state(cell) for cell in self.rnn_fwd]
        init_carry_bw = [init_state(cell) for cell in self.rnn_bwd]

        def scan_fn(carry, x_t):
            carries_fw, carries_bw = carry
            input_fw = x_t
            input_bw = x_t[::-1] 

            new_carries_fw = []
            new_carries_bw = []
            for i in range(self.num_layers):
                # Forward
                carry_fw, out_fw = self.rnn_fwd[i](carries_fw[i], input_fw)
                if self.layer_norm_fwd[i] is not None:
                    out_fw = self.layer_norm_fwd[i](out_fw)
                out_fw = self.dropout_fwd[i](out_fw, rngs=nnx.Rngs({'dropout': dropout_rngs}))
                new_carries_fw.append(carry_fw)
                input_fw = out_fw

                # Backward
                carry_bw, out_bw = self.rnn_bwd[i](carries_bw[i], input_bw)
                if self.layer_norm_bwd[i] is not None:
                    out_bw = self.layer_norm_bwd[i](out_bw)
                out_bw = self.dropout_bwd[i](out_bw, rngs=nnx.Rngs({'dropout': dropout_rngs}))
                new_carries_bw.append(carry_bw)
                input_bw = out_bw

            return (new_carries_fw, new_carries_bw), (input_fw, input_bw)

        _, (outputs_fw, outputs_bw) = jax.lax.scan(scan_fn, (init_carry_fw, init_carry_bw), x)

        outputs = jnp.concatenate([outputs_fw, outputs_bw[::-1]], axis=-1)
        return outputs

class RNN(nnx.Module):
    """
    Multi-layer (unidirectional) RNN encoder with optional layer norm and dropout.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        time_dim:int,
        control_dim:int, 
        num_layers: int,
        rngs: nnx.Rngs,
        modeltype="simple",
        dropout=0.0,
        use_layernorm=False,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Base input feature dimension (excluding time/control features). The actual
            expected per-timestep feature dimension is `input_dim + time_dim + control_dim`.

        hidden_dim : int
            Hidden size for each recurrent layer.

        time_dim : int
            Size of per-timestep time features concatenated to inputs.

        control_dim : int
            Size of per-timestep control/stimulus features concatenated to inputs.

        num_layers : int
            Number of recurrent layers.

        rngs : nnx.Rngs
            RNG container for parameter initialization.

        modeltype : {"simple", "gru", "lstm"}, default="simple"
            Recurrent cell type.

        dropout : float, default=0.0
            Dropout rate applied after each layer output.

        use_layernorm : bool, default=False
            Whether to apply LayerNorm after each layer output.
        """
        self.time_dim = time_dim
        self.input_dim = input_dim + time_dim + control_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.modeltype = modeltype
        self.use_layernorm = use_layernorm

        self.cells = []
        self.dropouts = []
        self.layernorms = []

        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            if modeltype == "lstm":
                cell = nnx.nn.recurrent.OptimizedLSTMCell(in_dim, hidden_dim, rngs=rngs)
            elif modeltype == "gru":
                cell = nnx.nn.recurrent.GRUCell(in_dim, hidden_dim, rngs=rngs)
            else:
                cell = nnx.nn.recurrent.SimpleCell(in_dim, hidden_dim, rngs=rngs)

            self.cells.append(cell)
            self.dropouts.append(Dropout(dropout, rngs=rngs))
            if use_layernorm:
                self.layernorms.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
            else:
                self.layernorms.append(None)

    def init_state(self, x_dtype):
        if self.modeltype == "lstm":
            return [
                (
                    jnp.zeros((self.hidden_dim,), dtype=x_dtype),
                    jnp.zeros((self.hidden_dim,), dtype=x_dtype),
                )
                for _ in range(self.num_layers)
            ]
        else:
            return [
                jnp.zeros((self.hidden_dim,), dtype=x_dtype)
                for _ in range(self.num_layers)
            ]

    def __call__(self, x, dropout_rngs):
        def scan_fn(carries, x_t):
            current_input = x_t
            new_carries = []

            for i, (cell, carry, dropout, ln) in enumerate(
                zip(self.cells, carries, self.dropouts, self.layernorms)
            ):
                carry, out = cell(carry, current_input)
                if ln is not None:
                    out = ln(out)
                out = dropout(out, rngs=nnx.Rngs({'dropout': dropout_rngs}))
                current_input = out
                new_carries.append(carry)

            return new_carries, current_input  

        initial_carry = self.init_state(x.dtype)

        _, outputs = jax.lax.scan(scan_fn, initial_carry, x)
        return outputs  


### 
# TCN
####

class TCNBlock(nnx.Module):
    """
    Single residual Temporal Convolutional Network (TCN) block with causal dilated convolutions.
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float, rngs: nnx.Rngs):
        """
        Parameters
        ----------
        n_inputs : int
            Input channel dimension.

        n_outputs : int
            Output channel dimension.

        kernel_size : int
            Kernel size for the dilated causal convolutions.

        dilation : int
            Dilation factor for the convolutions.

        dropout : float
            Dropout rate applied after each convolution + normalization.

        rngs : nnx.Rngs
            RNG container for parameter initialization.
        """
        self.conv1 = nnx.Conv(n_inputs, n_outputs, kernel_size=kernel_size, kernel_dilation=dilation, padding='CAUSAL', rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.ln1 = nnx.LayerNorm(n_outputs, rngs=rngs)
        self.conv2 = nnx.Conv(n_outputs, n_outputs, kernel_size=kernel_size, kernel_dilation=dilation, padding='CAUSAL', rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.ln2 = nnx.LayerNorm(n_outputs, rngs=rngs)
        self.downsample = nnx.Conv(n_inputs, n_outputs, kernel_size=1, rngs=rngs) if n_inputs != n_outputs else None

    def __call__(self, x):
        out = self.conv1(x)
        out = nnx.tanh(out)
        out = self.ln1(out)
        out = self.dropout1(out)        
        out = self.conv2(out)
        out = self.ln2(out)
        out = self.dropout2(out)
        out = nnx.tanh(out)
        res = x if self.downsample is None else self.downsample(x)
        
       
        return nnx.tanh(out + res)

class TCN(nnx.Module):
    """
    Stack of `TCNBlock`s with exponentially increasing dilation.

    Dilation for block i is `2 ** i`, giving a rapidly expanding receptive field.
    """
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 2, dropout: float = 0, rngs: nnx.Rngs = None):
        self.layers = []
        for i, out_channels in enumerate(num_channels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            block = TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout, rngs)
            self.layers.append(block)
       

    def __call__(self, x):
        for block in self.layers:
            x = block(x) 
        return x

class TCNEncoder(nnx.Module):
    def __init__(self, dim_in: int, dim_hidden: int, n_blocks: int, kernel_size: int, dropout: float, rngs: nnx.Rngs):
        self.tcn = TCN(
            num_inputs=dim_in,
            num_channels=[dim_hidden] * n_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
            rngs=rngs
        )

    def __call__(self, x):
        out = self.tcn(x)
        return out[-1] 

 