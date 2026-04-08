from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import jax.random as jrandom
from copy import deepcopy
import logging
from utils.timeseries import generate_ar_timeseries, generate_normal_timeseries
import utils.model_saving as ms

logging.basicConfig(level=logging.INFO)

class FlowTrainer(ABC):
    """
    Abstract base class for training flow-based models with JAX/Flax.
    Handles optimization, evaluation, and early stopping logic.
    """

    def __init__(self, 
                 flow_model, 
                 train_loader, 
                 val_loader, 
                 rngs,
                 runconfig,
                 trainable_params,
                 patience: int = 10,
                 log_interval: int = 100):
        """
        Initialize the trainer.

        Args:
            flow_model: Flax model to be trained.
            train_loader: Iterable training data loader.
            val_loader: Iterable validation data loader.
            rngs: JAX RNG sequence or callable.
            runconfig: Configuration object with training hyperparameters.
            trainable_params: Optional parameter filter for partial training.
            patience: Early stopping patience (in evaluation steps).
            log_interval: Frequency (in steps) for logging and validation.
        """
        self.flow_model = flow_model
        self.best_model = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rngs = rngs
        self.modeltype = runconfig.modeltype
        if trainable_params is not None: 
            self.optimizer = nnx.Optimizer(flow_model, optax.adam(runconfig.lr), wrt=trainable_params )
        else:
            self.optimizer = nnx.Optimizer(flow_model, optax.adam(runconfig.lr) )
        self.training_steps = runconfig.epochs
        self.patience = patience
        self.log_interval = log_interval
        self.out_folder = runconfig.out_folder
        self.trainable_params=trainable_params
        self.context_length_steps = runconfig.context_length_steps
        

    @abstractmethod
    def loss_fn(self, model, batch, rngs):
        pass

    @nnx.jit(static_argnums=(0,))
    def train_step(self, model, optimizer, batch, rngs):
        """
        Perform a single optimization step.

        Args:
            model: Flow model.
            optimizer: Optimizer instance.
            batch: Training batch.
            rngs: JAX RNGs.

        Returns:
            Scalar training loss.
        """
        if self.trainable_params is not None: 
            diff_state = nnx.DiffState(0, self.trainable_params) # filter head params of the first argument
            loss, grads = nnx.value_and_grad(self.loss_fn, argnums=diff_state)(model, batch, rngs)

        else:
            loss, grads = nnx.value_and_grad(self.loss_fn)(model, batch, rngs)
        
        optimizer.update(grads)
        return loss

    def evaluate(self, model):
        """
        Evaluate the model on the validation set.

        Args:
            model: Flow model.

        Returns:
            Mean validation loss.
        """
        val_losses = []
        for batch in self.val_loader:
            val_loss = self.loss_fn(model, batch, self.rngs)
            val_losses.append(val_loss)
        return float(jnp.mean(jnp.array(val_losses)))

    def train(self):
        """
        Run the full training loop with early stopping.

        Returns:
            best_model: Best-performing model (by validation loss).
            train_losses: List of logged training losses.
            val_losses: List of logged validation losses.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        logging.info("Training started ..")
        for step in range(self.training_steps):
            batch = next(iter(self.train_loader))
            train_loss = self.train_step(self.flow_model, self.optimizer, batch, self.rngs)

            if step % self.log_interval == 0:
                val_loss = self.evaluate(self.flow_model)
                logging.info(f"Step {step}: Train Loss: {train_loss}, Val Loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model = deepcopy(self.flow_model)
                    patience_counter = 0
                    # if self.out_folder:
                    #     ms.save_model(self.best_model, f"{self.out_folder}/checkpoints")
                else:
                    patience_counter += 1

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if patience_counter > self.patience:
                    logging.info(f"Early stopping at step {step}")
                    break
        logging.info("Training finished.")
        return self.best_model, train_losses, val_losses

class StandardFlowTrainer(FlowTrainer):
    """
    Trainer for standard flow matching.
    """

    def loss_fn(self, model, batch, rngs):
        """
        MSE loss

        Args:
            model: Flow model.
            batch: Tuple of conditioning and future observations.
            rngs: JAX RNGs.

        Returns:
            Scalar loss value.
        """
        x1_l, u, x1_f = batch  
        B, f, D = x1_f.shape
        t = jrandom.uniform(rngs(), (B,), minval=0.0, maxval=1.0)
        x0_keys = jrandom.split(rngs(), B)
        x0 = jax.vmap(generate_normal_timeseries, in_axes=(0, None, None))(x0_keys, D, f)  
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1_f 
        target_flow = x1_f - x0 
        pred_flow = model(x_t, x1_l, u, t, rngs) if self.modeltype == "patchtst" else jax.vmap(model, in_axes=(0,0,0,0, None))(x_t, x1_l, u, t, rngs())
        
        return jnp.mean((pred_flow - target_flow) ** 2)

class ARFlowTrainer(FlowTrainer):
    """
    Trainer for autoregressive flow matching models.
    """
    def loss_fn(self, model, batch, rngs):
        """
        MSE loss

        Args:
            model: Flow model.
            batch: Tuple of conditioning and future observations.
            rngs: JAX RNGs.

        Returns:
            Scalar loss value.
        """
        past_obs, control, next_obs, subject_id = batch
        x1 = next_obs
        t = jrandom.uniform(rngs(), (x1.shape[0],), minval=0.0, maxval=1.0)
        x0 = jrandom.normal(rngs(), x1.shape)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        pred_flow = model(x_t, past_obs, control, t, subject_id) if self.modeltype == "patchtst" else jax.vmap(model, in_axes=(0,0,0,0,0, None))(x_t, past_obs, control, t, subject_id, rngs())
        return jnp.mean(jnp.square(pred_flow - (x1 - x0)))
