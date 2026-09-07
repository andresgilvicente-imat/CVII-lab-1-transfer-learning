"""Training and evaluation transforms for Oxford-IIIT Pet images."""

from torchvision import transforms

from src.config import config


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Build ImageNet-compatible train transforms with data augmentation.
    The pipeline is: random resize + random horizontal flip + random rotation +
    color jitter + transform to tensor + normalization + random erasing.

    Args:
        image_size: Size to which the image is reshaped.

    Returns:
        Train transformation.
    """

    # TODO


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Build deterministic ImageNet-compatible validation and test transforms.
    The pipeline is the same as before, but without the random operations.

    Args:
        image_size: Size to which the image is reshaped.

    Returns:
        Eval transformation.
    """

    # TODO
