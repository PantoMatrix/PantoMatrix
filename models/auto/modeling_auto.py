# modeling_auto.py
from collections import OrderedDict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from ...models.camn_audio.modeling_camn_audio import CamnAudioModel
from ...models.camn_audio.configuration_camn_audio import CamnAudioConfig

logger = logging.get_logger(__name__)

MODEL_MAPPING = OrderedDict(
    [
        (CamnAudioConfig, CamnAudioModel),
        # Add other model mappings here if needed
    ]
)


class AutoModel:
    r"""
    AutoModel is a generic model class to instantiate a model from a configuration.
    It is designed to be instantiated using the `from_pretrained` method:
        model = AutoModel.from_pretrained("model_name_or_path")
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from transformers import PretrainedConfig
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in MODEL_MAPPING.keys())}."
        )
