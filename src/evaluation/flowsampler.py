from abc import ABC, abstractmethod
import jax.numpy as jnp
import flax.nnx as nnx
import jax 
import jax.random as jrandom
import diffrax as dfx
from diffrax import ODETerm
import numpy as np
import logging
from jax import lax
import zipfile, os
import utils.timeseries as timeseries

logging.basicConfig(level=logging.INFO)

class FlowSampler(ABC):
    """
    Abstract base class for generating flow-based model predictions.

    This class standardizes:
      - managing model + dataloaders
      - collecting per-episode predictions and concatenating them
      - saving mean predictions (and optionally full sample distributions)

    Subclasses must implement:
      - `sample(...)`: the sampler's core forward-generation routine
      - `_sample_batch(batch, key)`: how to turn one batch into `(batch_samples, batch_mean)`
    """

    def __init__(self, 
                 model, 
                 data_loaders, 
                 sampling_type, 
                 fmri_frame_counts, 
                 runconfig):
        """
        Parameters
        ----------
        model : Callable
            The generative model / vector field used by the sampler.

        data_loaders : Sequence[Iterable]
            A sequence of batch iterables. Ordering should correspond to
            `fmri_frame_counts.items()`.

        sampling_type : str
            Identifier used for naming output files (e.g., "standard", "ar", "lnsde").
            Saved predictions will include this string in their filename.

        fmri_frame_counts : Mapping[Any, int]
            Mapping from episode identifier -> number of frames 

        runconfig : Any
            Configuration object
        """
        self.model = model
        self.data_loaders = data_loaders
        self.fmri_frame_counts = fmri_frame_counts
        self.sampling_type = sampling_type
        self.subject_id = runconfig.subject_id
        self.save_distribution = runconfig.save_distribution

        self.n_samples = 100
        self.n_steps_flow = 100
        self.dim_y = runconfig.dim_y
        
        self.file_path = f'{runconfig.out_folder}/preds_{sampling_type}.npy'
        self.file_path_samples = f'{runconfig.out_folder}/preds_{sampling_type}_samples.npy'
        self.zip_file = f'{runconfig.out_folder}/fmri_predictions_friends_s7.zip'
    

    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        Generate samples from the flow model.

        Subclasses define the signature, but the method should return sample predictions
        produced by integrating the model's vector field (or equivalent) forward.
        """
        pass

    @abstractmethod
    def _sample_batch(self, batch, key):
        """
        Sample predictions for a single batch and compute the sample mean.

        Parameters
        ----------
        batch : Any
            A single batch produced by a loader. The contents depend on the subclass/
            dataset (e.g., (x_history, u, x_true) or (x_context, u_context, x_next, u_next)).

        key : jax.random.PRNGKey
            PRNG key used to generate randomness for this batch. `collect_predictions()`
            deterministically assigns a unique key per batch.

        Returns
        -------
        batch_preds : jax.Array
            Per-sample predictions for this batch (i.e., includes the `n_samples` axis).

        mean_preds : jax.Array
            Mean prediction over the sample axis for this batch.

        Notes
        -----
        The returned arrays must be compatible with the concatenation logic
        in `collect_predictions()`.
        """
        pass

    def collect_predictions(self):
        """
        Iterate over all episodes and their loaders, sample predictions batch-by-batch,
        concatenate them into episode-level arrays, and return
        the mean predictions.

        Returns
        -------
        dict
            Mapping `episode -> mean_predictions` after truncation to the configured
            number of frames for that episode.
        """
        preds_dict = {}

        preds_dict_samples = {}

        num_batches = sum(len(loader) for loader in self.data_loaders)
        master_key = jrandom.PRNGKey(0)
        batch_keys = jrandom.split(master_key, num_batches)

        batch_idx = 0
        for i, (epi, steps) in enumerate(self.fmri_frame_counts.items()):
            epi_preds_raw = []
            epi_preds_mean = []

            # Collect predictions for episode
            for batch in self.data_loaders[i]:
                key = batch_keys[batch_idx]
                
                batch_preds, mean_preds = self._sample_batch(batch, key)
                epi_preds_mean.append(mean_preds)
                epi_preds_raw.append(batch_preds)  # keep samples
                batch_idx += 1

            epi_pred_mean = jnp.concatenate(jnp.concatenate(epi_preds_mean, axis=0), axis=0)
            epi_pred_raw = jnp.concatenate(epi_preds_raw, axis=0)
            epi_pred_raw = epi_pred_raw.transpose(0, 2, 1, 3).reshape(-1, epi_pred_raw.shape[1], epi_pred_raw.shape[-1])
            del epi_preds_mean, epi_preds_raw
            logging.info(f"Epi {epi} done")
            preds_dict[epi] = epi_pred_mean[:steps]
            preds_dict_samples[epi] = epi_pred_raw[:steps]
            del epi_pred_mean, epi_pred_raw
            
        try:
            np.save(self.file_path, preds_dict, allow_pickle=True)
            if self.save_distribution:
                np.save(self.file_path_samples, preds_dict_samples, allow_pickle=True)
            print("Successfully saved")

        except: 
            print("no valid path")

        return preds_dict

class ARFlowSampler(FlowSampler):
    """
    Autoregressive flow sampler.

    This sampler predicts a sequence of length `n_steps_pred` by iteratively generating
    the next step from a rolling context window, updating the context each
    step. Sampling is repeated `n_samples` times and can be performed either:
      - per-example (via `sample`)
    """
    def __init__(self, model, data_loaders, sampling_type, fmri_frame_counts, runconfig):
        """
        Parameters
        ----------
        model, data_loaders, sampling_type, fmri_frame_counts, runconfig
            See `FlowSampler.__init__`.

        Notes
        -----
        Expects `runconfig` to provide:
          - segment_length_steps : int, steps in future segment
          - modeltype : str, used to select encoder
        """
        super().__init__(model, data_loaders, sampling_type, fmri_frame_counts, runconfig)
        
        self.n_steps_pred = runconfig.segment_length_steps
        self.modeltype = runconfig.modeltype
    
    @nnx.jit(static_argnums=(0))
    def sample(self, keys, history, covariates):
        """
        Sample trajectory

        Parameters
        ----------
        keys : jax.random.PRNGKey
            PRNG key used to generate `n_samples` independent sample trajectories.

        history : jax.Array
            Context window used for conditioning. Shape is typically
            (context_length, dim_y).

        covariates : jax.Array
            Covariate sequence aligned with the context/prediction horizon.

        Returns
        -------
        jax.Array
            Sample predictions.
        """
        @nnx.jit
        def predict(key, history, covariates):
            context_length = history.shape[0]
            step_keys = jrandom.split(key, self.n_steps_pred)

            def vf(t, x, args):
                history, covariates, key = args
                return self.model(x, history, covariates, t, self.subject_id, key)

            term = ODETerm(vf)
            solver = dfx.Tsit5()
            dt = 1 / self.n_steps_flow

            def scan_step(carry, inputs):
                current_context, key = carry
                i = inputs

                x0 = jrandom.normal(key, (self.dim_y))
                cov = lax.dynamic_slice(covariates, (i, 0), (context_length, covariates.shape[1]))

                x_next = dfx.diffeqsolve(
                    term,
                    solver,
                    0., 1.,
                    dt,
                    x0,
                    args=(current_context, cov, key),
                    saveat=dfx.SaveAt(t1=True)
                ).ys

                new_context = jnp.concatenate([current_context, x_next], axis=0)[-context_length:]
                return (new_context, step_keys[i]), x_next

            (_, _), predictions = lax.scan(
                scan_step,
                init=(history, step_keys[0]),  # key will be updated in scan
                xs=jnp.arange(self.n_steps_pred)
            )

            return jnp.swapaxes(predictions, 0, 1)  

        key_samples = jrandom.split(keys, self.n_samples)
        vmapped_predict = jax.vmap(predict, in_axes=(0, None, None))
        pred_samples = vmapped_predict(key_samples, history, covariates)
        return pred_samples
    
    def _sample_batch(self, batch, key):
        """
        Unpack a batch, generate sample predictions, and compute the mean prediction.

        Parameters
        ----------
        batch : tuple
            Expected to be (x_past, u, x_true).

        key : jax.random.PRNGKey
            Batch PRNG key.

        Returns
        -------
        batch_preds : jax.Array
            Per-sample predictions for the batch.

        mean_preds : jax.Array
            Sample mean predictions for the batch.
        """
        x_past, u, _ = batch
        batch_preds = self._collect_predictions_core(x_past, jnp.array(u), key)
                
        mean_preds = jnp.mean(batch_preds, axis=1)
        return jnp.squeeze(batch_preds, axis=2) , jnp.squeeze(mean_preds, axis=1)
    
    def _collect_predictions_core(self, x_past,u, key):
        """
        Dispatch sampling.

        Parameters
        ----------
        x_past : jax.Array
            Past/context observations.

        u : jax.Array
            Covariates.

        key : jax.random.PRNGKey
            PRNG key for this batch.

        Returns
        -------
        jax.Array
            Raw sample predictions. 
        """
        
        batch_size = int(jnp.shape(u)[0])
        keys = jrandom.split(key, batch_size)
        vmapped_sample = jax.vmap(self.sample, in_axes=(0,0, 0))
        preds = vmapped_sample(keys, x_past, u)

        return preds

    
class StandardFlowSampler(FlowSampler):
    """
    Standard flow sampler.

    """
    def __init__(self, model, data_loaders, sampling_type, fmri_frame_counts, runconfig):
        """
        Parameters
        ----------
        model, data_loaders, sampling_type, fmri_frame_counts, runconfig
            See `FlowSampler.__init__`.

        Notes
        -----
        Expects `runconfig` to provide:
          - ts : array-like, time grid for initializing the latent/initial series
          - context_length_steps : int
          - segment_length_steps : int
          - hrf_delay : int
          - modeltype : str, used to choose encoder model
        """
        super().__init__(model, data_loaders, sampling_type, fmri_frame_counts, runconfig)
        
        self.ts = runconfig.ts
        self.patch_len = runconfig.context_length_steps + runconfig.segment_length_steps + runconfig.hrf_delay
        self.modeltype = runconfig.modeltype
        self.context_length_steps = runconfig.context_length_steps

    @nnx.jit(static_argnums=(0))   
    def sample(self, key, x_history, u):
        """
        Sample trajectories.

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key used to generate `n_samples` independent trajectories.

        x_history : jax.Array
            Conditioning history (context).

        u : jax.Array
            Covariates aligned with the time horizon.

        Returns
        -------
        jax.Array
            Sample trajectories
        """
        @nnx.jit
        def predict(key, n_steps_flow):
            def vf(t, x, args):
                x_history, u = args
                
                return self.model(x, x_history, u,  jnp.array([t]), key)
                
            term = ODETerm(vf)
            solver = dfx.Tsit5()
            dt = 1/n_steps_flow
            x0 = timeseries.generate_normal_timeseries(key, self.dim_y, len(self.ts))

            predictions = dfx.diffeqsolve(term, solver, 0., 1., dt, x0, saveat=dfx.SaveAt(t1=True), args=(x_history,u)).ys
            return predictions

        key_samples = jrandom.split(key, self.n_samples)
        vmapped_predict = jax.vmap(predict, in_axes=(0, None)) 
        
        samples = vmapped_predict(key_samples, self.n_steps_flow)
            
            
        return jnp.squeeze(samples, axis=1)
    
    def _sample_batch(self, batch, key):
        """
        Unpack a batch, generate sample predictions, and compute the mean prediction.

        Parameters
        ----------
        batch : tuple
            Expected to be (x_history, u, x_f). 

        key : jax.random.PRNGKey
            Batch PRNG key.

        Returns
        -------
        batch_preds : jax.Array
            Per-sample predictions for the batch.

        mean_preds : jax.Array
            Mean prediction over the sample axis for the batch.
        """

        x_history, u, _ = batch
       
        batch_preds = self._collect_predictions_core(x_history, u, key)

        mean_preds = jnp.mean(batch_preds, axis=1)
        return batch_preds, mean_preds
    
    def _collect_predictions_core(self,x_history, U, key):
        """
        Vectorize `sample` across examples in the batch.

        Parameters
        ----------
        x_history : jax.Array
            Batched history/context.

        U : jax.Array
            Batched covariates.

        key : jax.random.PRNGKey
            PRNG key for the batch

        Returns
        -------
        jax.Array
            Per-example sample predictions
        """
        num_examples = jnp.shape(U)[0]
        keys = jrandom.split(key, num_examples)
        vmapped_sample = jax.vmap(self.sample, in_axes=(0,0, 0))
        preds = vmapped_sample(keys, x_history, U)
        
        return preds
    
    