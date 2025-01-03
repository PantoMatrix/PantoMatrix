# configuration_auto.py
from collections import OrderedDict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from ...models.camn_audio.configuration_camn_audio import CamnAudioConfig

logger = logging.get_logger(__name__)

CONFIG_MAPPING = OrderedDict(
    [
        ("camn_audio", CamnAudioConfig),
        # Add other model configurations here if needed
    ]
)


class AutoConfig:
    r"""
    AutoConfig is a generic configuration class to instantiate a model configuration.
    It is designed to be instantiated using the `from_pretrained` method:
        config = AutoConfig.from_pretrained("model_name_or_path")
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import PretrainedConfig
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict:
            model_type = config_dict["model_type"]
            if model_type in CONFIG_MAPPING:
                config_class = CONFIG_MAPPING[model_type]
                return config_class.from_dict(config_dict, **kwargs)
            else:
                raise ValueError(f"Unrecognized model type {model_type} in config.json.")
        else:
            # Fallback: If no model_type, try all config classes and see if any matches
            for pattern, config_class in CONFIG_MAPPING.items():
                if pattern in pretrained_model_name_or_path:
                    return config_class.from_dict(config_dict, **kwargs)
            # If still not found
            raise ValueError("Config file is missing the `model_type` field and no default model type could be inferred.")
