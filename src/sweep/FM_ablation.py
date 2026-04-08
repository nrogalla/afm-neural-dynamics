import optuna
import wandb
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.flowmatching import train_fm, compute_model_correlation, get_model
from data.dataloaders import load_subject_data, create_dataloaders
from utils.runconfigs import SFMRunConfigs, LSDERunConfigs, AFMRunConfigs

import logging
import argparse
import numpy as np
import gc
import psutil
import jax
import copy
import flax.nnx as nnx
from optuna.samplers import GridSampler

logging.basicConfig(level=logging.INFO)

ABLATION_SPACE = {
    "segment_length_in_s": [x for x in range(0, 30, 5) if x not in [10]],
    "context_length_in_s": [x for x in range(0, 51, 5) if x not in [30]]
}


def create_objective(runconfig, ablation_type, train_loader, val_loader, test_loader, train_sampling_loader, data, STUDY_NAME):  
    def objective(trial):
        trial_runconfig = copy.deepcopy(runconfig)
        try:
            param_name = f"{ablation_type}_length_in_s"

            length = trial.suggest_categorical(param_name, ABLATION_SPACE[param_name])
            
            trial_runconfig.out_folder = os.path.join(
                "outputs_ablation",
                f"{param_name}_{length}"
            )

            if trial_runconfig.out_folder is not None:
                os.makedirs(trial_runconfig.out_folder, exist_ok=True)
    
            CONFIG = {
                f"{param_name}" : length
            }
            

            config = dict(trial.params)
            config["trial.number"] = trial.number
            wandb.init(
                project=STUDY_NAME,                
                config=config,
                group=STUDY_NAME,
                reinit=True,
            )
            logging.info(trial.params)
            trial_runconfig.apply_config(CONFIG)
            trial_runconfig.print_config()

            # Training
            rngs = nnx.Rngs(0)
            model = get_model(trial_runconfig, rngs)
            
            losses, val_losses, best_model = train_fm(model, train_loader, val_loader, runconfig = trial_runconfig, rngs=rngs)
        
            
            for i in range(len(losses)):
                wandb.log(data={"train loss": losses[i]}, step=i)
                trial.report(val_losses[i], i)
                wandb.log(data={"val loss": val_losses[i]}, step=i)

            if trial.should_prune():
                logging.info("Pruned")
                wandb.finish(quiet=True)
                raise optuna.TrialPruned()
            

            val_corr, _ = compute_model_correlation(trial_runconfig, best_model, test_loader, train_sampling_loader, data)
            
            wandb.run.summary["val_corr"] = val_corr
            wandb.run.summary["train_loss"] = np.min(losses)
            wandb.run.summary["val_loss"] = np.min(val_losses)
            
        finally:
            wandb.finish(quiet=True)
            del trial_runconfig, model, best_model, losses, val_losses
            
            gc.collect()
            jax.clear_caches()
        
        logging.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
        return val_corr
    return objective

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run flow matching training with different configurations.")
    parser.add_argument('--framework', type=str, required=True, choices=['latentsde', 'sfm', 'afm'],
                        help='The modeling framework to use.')
    parser.add_argument('--modeltype', type=str, required=True,
                        help='The model type (e.g., "tcn", "mlp", etc.).')
    parser.add_argument('--area', type=str, default=None,
                        help='Brain area to use (e.g., "V1"). Optional.')
    parser.add_argument('--modality', type=str, default="all",
                        help='Stimulus modality to use (e.g., "visual"). Optional.')

    parser.add_argument('--drift_type', type=str, default="Neural",
                        help='Only applicable for LSDE framework. Choose drift type from "Neural", "Bistable", "Coupled" and "MultiCoupled". Optional.')
    parser.add_argument(
                        '--use_optimized',
                        action='store_true',
                        help='Flag to use tuned hyperparameters.'
                    )
    parser.add_argument(
                        '--ablation_type', type = str, default = "context",
                        help='Ablation study to run ("context" for ablation on context window, "segment" for ablation on prediction window).'
                    )

    args = parser.parse_args()

    if args.framework == "latentsde":
        runconfig = LSDERunConfigs(args.modeltype, drift_type = args.drift_type)

    elif args.framework == "sfm":
        runconfig = SFMRunConfigs(args.modeltype)

    elif args.framework == "afm":
        runconfig = AFMRunConfigs(args.modeltype, use_optimized=args.use_optimized)

    
    runconfig.out_folder = None
    runconfig.epochs = 20000
    runconfig.foundation = None
    runconfig.hyperopt = False

    data = load_subject_data(runconfig, modality=args.modality, area=args.area)

    if args.framework == "afm":
            train_loader, val_loader, test_loaders, train_sampling_loaders = create_dataloaders(
                data, runconfig, 1, framework=args.framework)
    else: 
            train_loader, val_loader, test_loaders, train_sampling_loaders = create_dataloaders(
                data, runconfig, runconfig.segment_length_steps, framework=args.framework)
    
    logging.info(f"Running optuna-wandb-{runconfig.framework}-{runconfig.modeltype}")
    os.environ["WANDB_API_KEY"] = "809a6a66fe39025c7bf69ea2b7f9ddfb8bf29ef4"
    wandb.login()

    os.environ["WANDB_START_METHOD"] = "thread"
  
    STUDY_NAME = f"{runconfig.framework}_{runconfig.modeltype}_ABLATION_{args.ablation_type}_len"

    # Run study

    sampler = GridSampler(search_space=ABLATION_SPACE)
    study = optuna.create_study(
                    direction="maximize",
                    study_name=STUDY_NAME,
                    sampler=sampler,
                    load_if_exists=True
                )
    logging.info(f"Sampler: {type(study.sampler).__name__}") 
    

    n_trials = len(ABLATION_SPACE[f"{args.ablation_type}_length_in_s"])

    objective = create_objective(runconfig, args.ablation_type, train_loader, val_loader, test_loaders, train_sampling_loaders, data, STUDY_NAME)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("Best trial:", study.best_trial.params)

