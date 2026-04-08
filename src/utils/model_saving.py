import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
from flax import nnx
import jax
from typing import Dict
import os
import orbax.checkpoint as ocp

from pathlib import Path

##########
# This circumvents orbax errors when saving LSTM models (Type Error: JAX array with PRNGKey dtype cannot be converted to a NumPy array)
# Code inspired by and adapted from https://github.com/google/flax/issues/4383
# when saving the model, rngs are converted to uint and reconverted to rng when loading the model from a saved checkpoint.
##########
def prepare_state_for_saving(model_state: Dict) -> Dict:
    """
    Converts RNG keys in the model state to arrays of uint32 numbers for saving.

    Args:
        model_state (Dict): The flattened model state.

    Returns:
        Dict: The updated model state with RNG keys converted.
    """
    updated_state = {}
    

    for key_path, var_state in model_state:  # Iterate over FlatState
        if var_state.type == nnx.RngKey:
            # Convert the RNG key into an array of uint32 numbers
            uint32_array = jax.random.key_data(var_state.value)
            # Replace the RNG key in the model state with the array
            new_var_state= nnx.VariableState(
                type=nnx.Param,  # Change the type to Param
                value=uint32_array
            )
            updated_state[key_path] = new_var_state
        else:
            # Keep other state variables unchanged
            updated_state[key_path]= var_state
    return updated_state


def flatstate_from_dict(state_dict: dict) -> nnx.statelib.FlatState:
    """
    Convert a dictionary into a FlatState object using `from_sorted_keys_values`.

    Args:
        state_dict (dict): A dictionary where keys are PathParts (tuples) and values are VariableState.

    Returns:
        FlatState: A FlatState object.
    """
    # Extract keys and values
    keys = tuple(sorted(state_dict.keys()))  # Ensure sorted order
    values = [state_dict[key] for key in keys]  # Maintain the same order

    # Use from_sorted_keys_values to create FlatState
    return nnx.statelib.FlatState.from_sorted_keys_values(keys, values)


def restore_rng_keys(model_state: Dict) -> Dict:
    """
    Converts arrays of uint32 numbers back to RNG keys in the model state.

    Args:
        model_state (Dict): The flattened model state.

    Returns:
        Dict: The updated model state with RNG keys restored.
    """

    updated_state = {}

    for key_path, var_state in model_state:  # Iterate over FlatState
        if var_state.type == nnx.Param and ('rngs' in key_path) and (key_path[-1] == 'key'):

            new_var_state = nnx.VariableState(
                type=nnx.RngKey,
                value=jax.random.wrap_key_data(var_state.value),

                tag='default',
            )
            updated_state[key_path] =new_var_state
        else:
            # Keep other state variables unchanged
            updated_state[key_path] = var_state


    return updated_state

def get_checkpoint_manager(path, clear = True):
    # Define the checkpoint directory
    absolute_path = os.path.abspath(path)
    # Create the directory if it doesn't exist
    os.makedirs(absolute_path, exist_ok=True)
    # Define CheckpointManagerOptions
    options = ocp.CheckpointManagerOptions(
        max_to_keep=2,
        keep_checkpoints_without_metrics=False,
        create=True,
    )
    if clear: 
    # Ensure the checkpoint directory is clean
        checkpoint_directory = ocp.test_utils.erase_and_create_empty(absolute_path)
    else:
        checkpoint_directory = Path(absolute_path)

    # Initialize the CheckpointManager
    checkpoint_manager = ocp.CheckpointManager(
        directory=checkpoint_directory,
        options=options,
    )
    return checkpoint_manager

def save_model(model, path): 
    _, state = nnx.split(model)
    # Flatten the model state
    model_state = state.flat_state()

    updated_state = prepare_state_for_saving(model_state)
    model_state_to_save = flatstate_from_dict(updated_state)
    
    checkpoint_manager = get_checkpoint_manager(path)
    # Save the checkpoint using StandardSave
    checkpoint_manager.save(
        step=0,
        args=ocp.args.StandardSave(nnx.State.from_flat_path(model_state_to_save)),
        force=True,
    )

    # Ensure checkpointing is finished
    checkpoint_manager.wait_until_finished()

def load_model_from_checkpoint(new_model, path):
    _, state = nnx.split(new_model)
    # Flatten the model state
    model_state = state.flat_state()

    updated_state = prepare_state_for_saving(model_state)
    model_state_to_save = flatstate_from_dict(updated_state)


    # Create a placeholder for the model state shape
    model_shape = nnx.eval_shape(lambda: nnx.State.from_flat_path(model_state_to_save))
    checkpoint_manager = get_checkpoint_manager(path, clear = False)
    # Restore the model state from the checkpoint
    restored_model_state = checkpoint_manager.restore(
        0,
        args=ocp.args.StandardRestore(model_shape),
    )

    # Flatten the restored model state
    restored_state_flat = restored_model_state.flat_state()

    # Convert arrays back to RNG keys
    restored_state_flat = flatstate_from_dict(restore_rng_keys(restored_state_flat))

    # Create nnx.State from the restored flat state
    restored_state = nnx.State.from_flat_path(restored_state_flat)

    # Update the model with the restored state
    nnx.update(new_model, restored_state)
    return new_model