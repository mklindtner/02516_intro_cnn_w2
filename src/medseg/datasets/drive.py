from pathlib import Path


from PIL import Image
import torch
import torchvision.transforms.v2 as transforms


DEFAULT_SPLITFILES_PATH = Path(__file__).parent / 'data' / 'DRIVE'
DEFAULT_BASE_PATH = Path('/dtu/datasets1/02516/DRIVE/training/')


DEFAULT_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)
DEFAULT_LABEL_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


def load_split_indices(path):
    path = Path(path)
    return path.read_text().splitlines()


def file_path(base, subdir, file):
    return Path(base) / subdir / file


class DRIVEDataset(torch.utils.data.Dataset):
    """DRIVE dataset

    Arguments:
        split: The split of data you want, i.e. 'train', 'test' or 'validation'.

    """
    def __init__(
        self,
        split,
        image_transform=DEFAULT_IMAGE_TRANSFORM,
        label_transform=DEFAULT_LABEL_TRANSFORM,
        image_dir='images',
        label_dir='mask',
        image_file_suffix='_training.tif',
        label_file_suffix='_training_mask.gif',
        splitfiles_path=DEFAULT_SPLITFILES_PATH,
        base_path=DEFAULT_BASE_PATH
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform

        indices = load_split_indices(Path(splitfiles_path) / (f'{split}.txt'))

        self.image_paths = [
            file_path(base_path, image_dir, f'{idx}{image_file_suffix}')
            for idx in indices
        ]
        self.label_paths = [
            file_path(base_path, label_dir, f'{idx}{label_file_suffix}')
            for idx in indices
        ]

    def __len__(self):
        """Return the total number of samples."""

        return len(self.image_paths)

    def __getitem__(self, idx):
        """Generate one sample of data."""

        image = Image.open(self.image_paths[idx])
        label = Image.open(self.label_paths[idx])

        X = self.image_transform(image)
        Y = self.label_transform(label)

        return X, Y
