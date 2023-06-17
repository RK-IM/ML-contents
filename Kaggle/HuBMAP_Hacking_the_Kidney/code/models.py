import torch
import segmentation_models_pytorch as smp

DECODERS = [
    'DeepLabV3',
    'DeepLabV3Plus',
    'FPN',
    'Linknet',
    'MAnet',
    'PAN',
    'PSPNet',
    'Unet',
    'UnetPlusPlus',
]
ENCODERS = list(smp.encoders.encoders.keys())


def define_model(
        decoder_name,
        encoder_name,
        num_classes=1,
        activation=None,
        encoder_weights='imagenet',
):
    """
    Define segmentation model using segmentation-models-pytorch,
    see 'https://github.com/qubvel/segmentation_models.pytorch'.

    Args:
        decoder_name (str): Name of architecture to use.
        encoder_name (str): Name of encoder to use.
        num_class (int): Number of classes to classify. Defaults to 1.
        activation (str, optional): Name of activation function, apply to last layer.
            Defaults to None,
        encoder_weights (str, optional): Pretrained model weights for encoder.
            Defaults to pretrained on "imagenet" dataset.
    """
    assert decoder_name in DECODERS, "Decoder name not supported"
    assert encoder_name in ENCODERS, "Encoder name not supported"

    decoder = getattr(smp, decoder_name)

    model = decoder(
        encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation
    )
    
    return model
