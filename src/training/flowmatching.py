import jax.numpy as jnp
import flax.nnx as nnx
import pickle
import logging
import evaluation.flowsampler as fs
import gc
import jax
logging.basicConfig(level=logging.INFO)

from evaluation.evaluation import compute_pearson_score_full
import models.latentsdemodel as lsfm
import models.autoregressiveflowmodel as afm
import models.standardflowmodel as sfm

from .flowtrainer import LNSDEFlowTrainer, ARFlowTrainer, StandardFlowTrainer 

def get_model(runconfig, rngs):
    """
    Construct a flow model instance based on the configured framework.
    """
   
    if runconfig.framework == "afm":
        flow_model = afm.Flow(runconfig = runconfig, rngs=rngs)
       
    elif runconfig.framework == "sfm":
        flow_model = sfm.Flow(runconfig = runconfig,rngs=rngs)
      
    else:
        logging.error("Framework not implemented")
    return flow_model

def train_fm(flow_model, train_loader, val_loader, runconfig, rngs, trainable_params = None):
    """
    Train a flow model using the trainer corresponding to the configured framework.
    """
    if runconfig.framework == "afm":
       
        trainer = ARFlowTrainer(flow_model, train_loader, val_loader,runconfig=runconfig, trainable_params=trainable_params, rngs=nnx.Rngs(rngs()))
    elif runconfig.framework == "sfm":
    
        trainer = StandardFlowTrainer(flow_model, train_loader, val_loader,runconfig=runconfig, trainable_params=trainable_params, rngs=nnx.Rngs(rngs()))
    else:
        logging.error("Framework not implemented")
    
    best_model, losses, val_losses = trainer.train()

    if runconfig.out_folder is not None: 
        with open(f'{runconfig.out_folder}/losses.pkl', 'wb') as file:
            pickle.dump(losses, file)
            pickle.dump(val_losses, file)
    
    return losses, val_losses, best_model

def compute_model_correlation(runconfig,model, test_loaders, train_sampling_loaders, data):
    """
    Generate predictions on a sampling split and compute mean Pearson correlation.
    """
    model.eval()
    def evaluate_preds(sampling_type, data_loaders):
        if runconfig.framework == "afm":
            flow_sampler = fs.ARFlowSampler(model=model, data_loaders=data_loaders, sampling_type=sampling_type,
                                fmri_frame_counts=data[sampling_type]["fmri_frame_counts"], runconfig=runconfig)
        
        elif runconfig.framework == "sfm":
            flow_sampler = fs.StandardFlowSampler(model=model, data_loaders=data_loaders, sampling_type=sampling_type,
                                fmri_frame_counts=data[sampling_type]["fmri_frame_counts"], runconfig=runconfig)
        
        else:
            logging.error("Framework not implemented")
        
        pred_dict = flow_sampler.collect_predictions()
        preds = []
        y_true = []
        for i, (epi, val) in enumerate(pred_dict.items()): 
            preds.append(val[5:-5, :])
            y_true.append(data[sampling_type]["fmri"][epi][5:-5, :])

        del pred_dict
        preds = jnp.concatenate(preds, axis=0)
        y_true = jnp.concatenate(y_true, axis=0)
        logging.info(f"Shape of predicted time series {jnp.shape(preds)}")
        logging.info(f"Shape of observed time series {jnp.shape(y_true)}")
        correlation = compute_pearson_score_full(y_true, preds)
        del preds, y_true
        gc.collect(); jax.clear_caches()
        return correlation
    
 
    test_correlation = evaluate_preds("test_sampling", test_loaders)
    logging.info(f"correlation test: {test_correlation}")

    # if not runconfig.hyperopt: 
    #     train_correlation = evaluate_preds("train_sampling", train_sampling_loaders)
    #     logging.info(f"correlation train: {train_correlation}")


    return test_correlation, None