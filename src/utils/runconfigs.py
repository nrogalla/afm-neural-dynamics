import jax.numpy as jnp
from .utils_combi import get_timesteps_from_sec
import diffrax as dfx
import logging
logging.basicConfig(level=logging.INFO)
import os
SHARED_DEFAULTS = {

    "lstm": {
        "hidden_layers": 2,
        "dim_hidden": 128,
        "batch_size": 128,
        "lr": 0.001,
        "dropout": 0.0
    },
    "gru": {
        "hidden_layers": 2,
        "dim_hidden": 128,
        "batch_size": 128,
        "lr": 0.001,
        "dropout": 0.0
    },
    "simple": {
        "hidden_layers": 2,
        "dim_hidden": 32,
        "batch_size": 64,
        "lr": 0.001,
        "dropout": 0.0
    },
    "tcn": {
        "hidden_layers": 3,
        "kernel_size": 4,
        "dim_hidden": 128,
        "batch_size": 64,
        "lr": 0.001,
        "dropout": 0.0
    }
}

OPTIMIZED = {
    "sfm": {

        "gru": {
            "hidden_layers": 3,
            "dim_hidden": 256,
            "batch_size": 64,
            "lr": 0.0005,
            "dropout": 0.0,
            "latent_blocks": 5,
            "latent_hidden": 128
        }
    },
    "afm": {
        "gru": {
            "latent_blocks": 5,
            "hidden_layers": 4,
            "latent_hidden": 16,
            "dim_hidden": 256,
            "batch_size": 64,
            "lr": 0.001,
            "dropout": 0.05
        }
    }
}

class BaseRunConfigs:
    """Holds run-wide configuration for dataset/model/training and output paths.
    """
    def __init__(self, modeltype: str, framework: str, use_optimized: bool = False):
        """Create a configuration object.

        Args:
            modeltype: Model identifier used to select defaults/overrides.
            framework: Framework name (e.g., "afm", "sfm") used for overrides and output naming.
            use_optimized: If True, apply tuned hyperparameter overrides after defaults.
        """
        self.movies_train =["friends-s01","friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures","movie10-life", "movie10-wolf"]
    
        self.movies_test = ["friends-s06"]
        self.root_dir = "" # change to directory of algonauts dataset
       
        self.modeltype = modeltype
        self.framework = framework
        self.use_optimized = use_optimized
        self.subject_id = 5
        self.epochs = 20000
        self.both = False
        self.context_length_in_s = 30
        self.segment_length_in_s = 10
        self.hrf_delay = 2
        self.hyperopt = False
        self.dim_t = 16
        self.save_distribution = False

        self.context_length_steps = get_timesteps_from_sec(self.context_length_in_s)
        self.segment_length_steps = get_timesteps_from_sec(self.segment_length_in_s)

        self.out_folder = os.path.join(f"outputs_{self.framework}_{self.modeltype}_s{self.subject_id}_stimonly")
        
        if self.out_folder is not None:
            os.makedirs(self.out_folder, exist_ok=True)
        self.apply_defaults()
        if self.use_optimized:
            self.apply_optimized()

    def apply_defaults(self):
        for key, val in SHARED_DEFAULTS.get(self.modeltype, {}).items():
            setattr(self, key, val)
    def apply_optimized(self):
        overrides = OPTIMIZED.get(self.framework, {}).get(self.modeltype, {})
        for key, val in overrides.items():
            setattr(self, key, val)
    
    def apply_config(self, config):
        for key, val in config.items():
            setattr(self, key, val)
            
        self.context_length_steps = get_timesteps_from_sec(self.context_length_in_s)
        self.segment_length_steps = get_timesteps_from_sec(self.segment_length_in_s)
        

    def print_config(self):
        logging.info("Setup")
        logging.info(f"Running {self.framework} {self.modeltype} with following specifications:")
        for attr, value in self.__dict__.items():
            logging.info(f"{attr}: {value}")


class AFMRunConfigs(BaseRunConfigs):
    def __init__(self, modeltype: str, use_optimized: bool = False):
        super().__init__(modeltype=modeltype, framework="afm", use_optimized=use_optimized)
        

class SFMRunConfigs(BaseRunConfigs):
    def __init__(self, modeltype: str, use_optimized: bool = False):
        super().__init__(modeltype=modeltype, framework="sfm", use_optimized=use_optimized)
       
        self.ts = jnp.arange(0, self.segment_length_in_s + 1.49, 1.49)