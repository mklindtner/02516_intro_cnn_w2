from .encdec import EncDec
from .unet01_skip import UNetSkip
from .unet01_dilated import DilatedNet


def list_models():
    return list(_MODELS.keys())


def get_model(name):
    return _MODELS[name]


_MODELS = {
    'EncDec': EncDec,
    'UNetSkip': UNetSkip,
    'DilatedNet': DilatedNet,
}
