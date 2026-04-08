import optuna
import wandb
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.flowmatching import train_fm, compute_model_correlation, get_model
from data.dataloaders import load_subject_data, load_multiple_subjects_data, create_dataloaders
from utils.runconfigs import SFMRunConfigs, LSDERunConfigs, AFMRunConfigs

import logging
import argparse
import numpy as np
from optuna.pruners import MedianPruner
import gc
import psutil
import jax
import copy
import flax.nnx as nnx
import jax.numpy as jnp
from optuna.samplers import GridSampler

logging.basicConfig(level=logging.INFO)

RUN_ABLATION = True
ABLATION_SPACE = {
    "segment_length_in_s": [x for x in range(0, 30, 5) if x not in [10]]#list(range(0, 25, 5)),
}
# ABLATION_SPACE = {
#     "context_length_in_s": [x for x in range(0, 51, 5) if x not in (20, 45)]
# }


def create_objective(runconfig, subjects, train_loader, val_loader, subject_test_loaders, subject_train_sampling_loaders, data_subjects, STUDY_NAME):  
    def objective(trial):
        trial_runconfig = copy.deepcopy(runconfig)
        try:
            if RUN_ABLATION:
                # context_length_in_s = trial.suggest_categorical("context_length_in_s", ABLATION_SPACE["context_length_in_s"])
                # trial_runconfig.out_folder = os.path.join("outputs_ablation", f"context_{context_length_in_s}")
                segment_length_in_s = trial.suggest_categorical("segment_length_in_s", ABLATION_SPACE["segment_length_in_s"])
                trial_runconfig.out_folder = os.path.join("outputs_ablation", f"pred_{segment_length_in_s}")
        
                if trial_runconfig.out_folder is not None:
                    os.makedirs(trial_runconfig.out_folder, exist_ok=True)
        
                CONFIG = {
                    # "context_length_in_s": context_length_in_s,
                    "segment_length_in_s" : segment_length_in_s
                }
            elif trial_runconfig.framework == "sfm" or trial_runconfig.framework == "afm":
                if trial_runconfig.modeltype == "gru":
                    lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3])
                    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
                    dim_hidden = trial.suggest_categorical("dim_hidden", [16, 32, 64, 128, 256, 512]) # not 1024 bc OOM kill
                    hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
                    latent_hidden = trial.suggest_categorical("latent_hidden", [8, 16, 32, 64, 128, 256, 512])
                    latent_blocks = trial.suggest_int("latent_blocks", 1, 6)
                    dropout = trial.suggest_categorical("dropout", [0, 0.05, 0.1, 0.2, 0.3])
                    
                CONFIG = {
                    "lr": lr,
                    "hidden_layers" : hidden_layers,
                    "dim_hidden" : dim_hidden,
                    "batch_size" : batch_size,
                    "latent_blocks" : latent_blocks,
                    "latent_hidden" : latent_hidden,
                    "dropout" : dropout
                }

            elif trial_runconfig.framework == "latentsde" and trial_runconfig.drift_type == "Neural":
                lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3])
                batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
                dim_hidden = trial.suggest_categorical("dim_hidden", [16, 32, 64, 128, 256, 512]) # not 1024 bc OOM kill
                hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
                drift_hidden = trial.suggest_categorical("drift_hidden", [16, 32, 64, 128, 256])
                drift_hidden_blocks = trial.suggest_int("drift_hidden_blocks", 1, 4)
                diffusion_hidden = trial.suggest_categorical("diffusion_hidden", [16, 32, 64, 128, 256])
                diffusion_hidden_blocks = trial.suggest_int("diffusion_hidden_blocks", 1, 4)
                dim_x = trial.suggest_categorical("dim_x", [16, 32, 64, 128, 256, 512])
                latent_dim_y = trial.suggest_categorical("latent_dim_y", [8, 16, 32])
                
                CONFIG = {
                    "drift_hidden" : drift_hidden,
                    "drift_hidden_blocks" : drift_hidden_blocks,
                    "diffusion_hidden" : diffusion_hidden,
                    "diffusion_hidden_blocks" : diffusion_hidden_blocks,
                    "hidden_layers" : hidden_layers,
                    "dim_hidden" : dim_hidden,
                    "dim_x" : dim_x,
                    "latent_dim_y" : latent_dim_y,
                    "lr": lr,
                    "batch_size": batch_size,
                }
            elif trial_runconfig.framework == "latentsde" and trial_runconfig.drift_type == "Coupled":
                lr = trial.suggest_categorical("learning_rate", [5e-4, 1e-3, 2e-3])
                drift_hidden = trial.suggest_categorical("drift_hidden", [32, 64, 128, 256])
                diffusion_hidden = trial.suggest_categorical("diffusion_hidden", [16, 32, 64, 128, 256])
                diffusion_hidden_blocks = trial.suggest_int("diffusion_hidden_blocks", 1, 4)
                dim_x = trial.suggest_categorical("dim_x", [16, 32, 64, 128, 256, 512])
                latent_dim_y = trial.suggest_categorical("latent_dim_y", [8, 16, 32, 64])
                
                CONFIG = {
                    "drift_hidden" : drift_hidden,
                    "diffusion_hidden" : diffusion_hidden,
                    "diffusion_hidden_blocks" : diffusion_hidden_blocks,
                    "dim_x" : dim_x,
                    "latent_dim_y" : latent_dim_y,
                    "lr": lr,
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
            shared_model = get_model(trial_runconfig, rngs)
            
            losses, val_losses, shared_model = train_fm(shared_model, train_loader, val_loader, runconfig = trial_runconfig, rngs=rngs)
        
            
            for i in range(len(losses)):
                wandb.log(data={"train loss": losses[i]}, step=i)

                trial.report(val_losses[i], i)
                wandb.log(data={"val loss": val_losses[i]}, step=i)

            if trial.should_prune():
                logging.info("Pruned")
                wandb.finish(quiet=True)
                raise optuna.TrialPruned()
            
            test_corr_per_subj = []
            head_folder = trial_runconfig.out_folder
            for subject_id in subjects:
                logging.info(f"Predicting for Subject {subject_id}")
                trial_runconfig.subject_id = subject_id
                test_loaders = subject_test_loaders[subject_id]
                train_sampling_loaders = subject_train_sampling_loaders[subject_id]
                data = data_subjects[subject_id]
                trial_runconfig.out_folder = f"{head_folder}/s{subject_id}"
                
                os.makedirs(runconfig.out_folder, exist_ok=True)
                test_corr, _ = compute_model_correlation(trial_runconfig, shared_model, test_loaders, train_sampling_loaders, data)
                test_corr_per_subj.append(test_corr)
                
            # Evaluation
            val_corr = jnp.mean(jnp.array(test_corr_per_subj))

            
            wandb.run.summary["val_corr"] = val_corr
            wandb.run.summary["train_loss"] = np.min(losses)
            wandb.run.summary["val_loss"] = np.min(val_losses)
            
            
            
        finally:
            wandb.finish(quiet=True)
            del trial_runconfig, shared_model, losses, val_losses
            
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

    args = parser.parse_args()

    if args.framework == "latentsde":
        runconfig = LSDERunConfigs(args.modeltype, drift_type = args.drift_type)

    elif args.framework == "sfm":
        runconfig = SFMRunConfigs(args.modeltype)

    elif args.framework == "afm":
        runconfig = AFMRunConfigs(args.modeltype, use_optimized=args.use_optimized)

    
    runconfig.out_folder = None

    subjects = [1, 2, 3, 5]
   
    if RUN_ABLATION:
        runconfig.out_folder = "outputs_ablation2"
        runconfig.epochs = 2#0000
        runconfig.foundation = None
        runconfig.hyperopt = False
        data = load_subject_data(runconfig, modality=args.modality, area=args.area)
        
    else:
        runconfig.epochs = 2000#15000
        runconfig.foundation = 1
        runconfig.hyperopt = True
        data = load_multiple_subjects_data(runconfig, subjects=subjects, modality=args.modality, area=args.area)

    if args.framework == "afm":
        train_loader, val_loader = create_dataloaders(
            data, runconfig, 1, framework=args.framework)
    else: 
        train_loader, val_loader = create_dataloaders(
            data, runconfig, runconfig.segment_length_steps, framework=args.framework)

    subject_test_loaders = {}
    subject_train_sampling_loaders = {}
    data_subjects = {}
    for subject_id in subjects:
        runconfig.subject_id = subject_id
        data = load_subject_data(runconfig, modality=args.modality, area=args.area)

        if args.framework == "afm":
            _, _, test_loaders, train_sampling_loaders = create_dataloaders(
                data, runconfig, 1, framework=args.framework)
            
        else: 
            _, _, test_loaders, train_sampling_loaders = create_dataloaders(
                data, runconfig, runconfig.segment_length_steps, framework=args.framework)
        subject_test_loaders[subject_id] = test_loaders
        subject_train_sampling_loaders[subject_id] = train_sampling_loaders
        data_subjects[subject_id] = data
    logging.info(f"Running optuna-wandb-{runconfig.framework}-{runconfig.modeltype}")
    os.environ["WANDB_API_KEY"] = "809a6a66fe39025c7bf69ea2b7f9ddfb8bf29ef4"
    wandb.login()

    os.environ["WANDB_START_METHOD"] = "thread"
    if runconfig.framework == "latentsde":
        STUDY_NAME = f"{runconfig.framework}_{runconfig.modeltype}_{runconfig.drift_type}_12_08"
    else:
        if RUN_ABLATION:

            STUDY_NAME = f"{runconfig.framework}_{runconfig.modeltype}_ABLATION_pred_len2"
        else:
            STUDY_NAME = f"{runconfig.framework}_{runconfig.modeltype}_08_08"


    # Run study
    if RUN_ABLATION:

        sampler = GridSampler(search_space=ABLATION_SPACE)
        study = optuna.create_study(
                        direction="maximize",
                        study_name=STUDY_NAME,
                        sampler=sampler,
                        load_if_exists=True
                    )
        logging.info(f"Sampler: {type(study.sampler).__name__}") 
        

    else:
        study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, pruner=MedianPruner(n_startup_trials=10))
  
    # n_trials = len(ABLATION_SPACE["context_length_in_s"]) if RUN_ABLATION else 100
    n_trials = len(ABLATION_SPACE["segment_length_in_s"]) if RUN_ABLATION else 100

    objective = create_objective(runconfig, subjects, train_loader, val_loader, subject_test_loaders, subject_train_sampling_loaders, data_subjects, STUDY_NAME)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("Best trial:", study.best_trial.params)

