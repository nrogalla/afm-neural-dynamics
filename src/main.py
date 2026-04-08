import argparse
import logging

from training.flowmatching import train_fm, compute_model_correlation, get_model
from data.dataloaders import load_subject_data, load_multiple_subjects_data, create_dataloaders
from utils.runconfigs import SFMRunConfigs, LSDERunConfigs, AFMRunConfigs
logging.basicConfig(level=logging.INFO)
import flax.nnx as nnx
import jax.numpy as jnp
import os

def main():
    # Command-line argument parsing
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

    parser.add_argument('--foundation', type=int, default=None,
                        help='Flag to train foundation model instead of individual model. 1: group model, ' \
                        '2: foundation model (train on 3, retrain decoder only on exluded, predict for excluded subject only), ' \
                        '3: foundation model (train on all, retrain decoder on all, predict for all), ' \
                        '4: foundation model (train on 3, retrain decoder and encoder layers only on excluded, predict for excluded subject only)')
    parser.add_argument(
                        '--use_optimized',
                        action='store_true',
                        help='Flag to use tuned hyperparameters.'
                    )

    args = parser.parse_args()
    
    if args.framework == "latentsde":
        runconfig = LSDERunConfigs(args.modeltype, drift_type = args.drift_type, use_optimized=args.use_optimized)

    elif args.framework == "sfm":
        runconfig = SFMRunConfigs(args.modeltype, use_optimized=args.use_optimized)

    elif args.framework == "afm":
        runconfig = AFMRunConfigs(args.modeltype, use_optimized=args.use_optimized)

    else:
        raise ValueError(f"Unsupported framework: {runconfig.framework}")
    
    runconfig.foundation = args.foundation
    runconfig.print_config()
    rngs = nnx.Rngs(0)
    
    if args.foundation is None: 
        data = load_subject_data(runconfig, modality=args.modality, area=args.area)
        
        if args.framework == "afm":
            train_loader, val_loader, test_loaders, train_sampling_loaders = create_dataloaders(
                data, runconfig, 1, framework=args.framework)
        else: 
            train_loader, val_loader, test_loaders, train_sampling_loaders =create_dataloaders(
                data, runconfig, runconfig.segment_length_steps, framework=args.framework)
        # Training
        
        model = get_model(runconfig, rngs)
        losses, val_losses, best_model = train_fm(model, train_loader, val_loader, runconfig = runconfig, rngs=rngs)

        # Evaluation
        compute_model_correlation(runconfig, best_model, test_loaders, train_sampling_loaders, data)
    else: 
        logging.info(f"Running args.foundation {args.foundation}")
        if args.foundation==1 or args.foundation == 3:
            subjects = [1, 2, 3, 5]
        elif args.foundation==2 or args.foundation == 4:
            subjects = [2,3,5]
            runconfig.training_subject_ids = subjects

        # Train on multiple subjects
        data = load_multiple_subjects_data(runconfig, subjects=subjects, modality=args.modality, area=args.area)

        if args.framework == "afm":
            train_loader, val_loader = create_dataloaders(
                data, runconfig, 1, framework=args.framework)
        else: 
            train_loader, val_loader = create_dataloaders(
                data, runconfig, runconfig.segment_length_steps, framework=args.framework)

        shared_model = get_model(runconfig, rngs)
            
        losses, val_losses, shared_model = train_fm(shared_model, train_loader, val_loader, runconfig = runconfig, rngs=rngs)
        del data, losses, val_losses, train_loader, val_loader
        if args.foundation==1 or args.foundation==3:
            # Predictions for all subjects
            test_corr_per_subj = []
            subjects = [2, 3,5]
            head_folder = runconfig.out_folder
            for subject_id in subjects:
                logging.info(f"Predicting for Subject {subject_id}")
                runconfig.subject_id = subject_id
                data = load_subject_data(runconfig, modality=args.modality, area=args.area)
        
                if args.framework == "afm":
                    train_loader, val_loader, test_loaders, train_sampling_loaders = create_dataloaders(
                        data, runconfig, 1, framework=args.framework)
                else: 
                    train_loader, val_loader, test_loaders, train_sampling_loaders = create_dataloaders(
                        data, runconfig, runconfig.segment_length_steps, framework=args.framework)
                
                if args.foundation == 3: # Retraining decoder
                    args.foundation_params = nnx.All(nnx.Param, nnx.PathContains('linear_out')) 

                    losses, val_losses, shared_model = train_fm(
                        shared_model,
                        train_loader,
                        val_loader,
                        runconfig=runconfig,
                        rngs=rngs,
                        trainable_params=args.foundation_params  
                    )
                del train_loader, val_loader
                runconfig.out_folder = f"{head_folder}/s{subject_id}"
                os.makedirs(runconfig.out_folder, exist_ok=True)
                test_corr, _ = compute_model_correlation(runconfig, shared_model, test_loaders, train_sampling_loaders, data)
                test_corr_per_subj.append(test_corr)
            
            logging.info(f"Mean correlation score: {jnp.mean(jnp.array(test_corr_per_subj))}")

        elif args.foundation==2 or args.foundation== 4:
            # Fixing Model and only training decoder for subj 1
            runconfig.subject_id = 1
            data = load_subject_data(runconfig, modality=args.modality, area=args.area)

            if args.framework == "afm":
                train_loader, val_loader, test_loaders, train_sampling_loaders = create_dataloaders(
                    data, runconfig, 1, framework=args.framework)
            else: 
                train_loader, val_loader, test_loaders, train_sampling_loaders = create_dataloaders(
                    data, runconfig, runconfig.segment_length_steps, framework=args.framework)
            
            if args.foundation== 2:
                args.foundation_params = nnx.All(nnx.Param, nnx.PathContains('linear_out'))
            else:
                args.foundation_params = nnx.All(
                                    nnx.Param,
                                    lambda path, node: any(nnx.PathContains(name) for name in ['linear_out', 'input_encoder', 'observation_encoder'])
                                )
                
            runconfig.epochs = 5000 # limited retraining
            losses, val_losses, _ = train_fm(
                shared_model,
                train_loader,
                val_loader,
                runconfig=runconfig,
                rngs=rngs,
                trainable_params=args.foundation_params  
            )
        
            compute_model_correlation(runconfig, shared_model, test_loaders, train_sampling_loaders, data)


                    
                
                    
                    

if __name__ == "__main__":
    main()
