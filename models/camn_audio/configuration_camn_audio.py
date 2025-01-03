from transformers import PretrainedConfig
from omegaconf import OmegaConf

class CamnAudioConfig(PretrainedConfig):
    model_type = "camn_audio"

    def __init__(self, config_obj=None, **kwargs):
        if config_obj is not None:
            cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
            kwargs.update(cfg_dict)

        super().__init__(**kwargs)
