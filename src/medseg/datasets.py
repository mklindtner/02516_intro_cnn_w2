from pathlib import Path


from PIL import Image
import torch
import torchvision.transforms.v2 as transforms


DEFAULT_SPLITFILES_PATH = Path(__file__).parent / 'data'
DEFAULT_BASE_PATH = Path('/dtu/datasets1/02516/PH2_Dataset_images/')


DEFAULT_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ]
)
DEFAULT_LABEL_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
    ]
)


def load_data_split(path):
    path = Path(path)
    return path.read_text().splitlines()


def image_path(base, name, dir_suffix='_Dermoscopic_Image', file_suffix='', ext='.bmp'):
    return Path(base) / name / (name + dir_suffix) / (name + ext)


def label_path(base, name, dir_suffix='_lesion', file_suffix='_lesion', ext='.bmp'):
    return Path(base) / name / (name + dir_suffix) / (name + file_suffix+ ext)


class PH2Dataset(torch.utils.data.Dataset):
    """PH2 dataset

    Arguments:
        split: The split of data you want, i.e. 'train', 'test' or 'validation'.

    """
    def __init__(
        self,
        split,
        image_transform=DEFAULT_IMAGE_TRANSFORM,
        label_transform=DEFAULT_LABEL_TRANSFORM,
        splitfiles_path=DEFAULT_SPLITFILES_PATH,
        base_path=DEFAULT_BASE_PATH,
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform

        datasets = load_data_split(Path(splitfiles_path) / (split + '.txt'))

        self.image_paths = [image_path(base_path, name) for name in datasets]
        self.label_paths = [label_path(base_path, name) for name in datasets]

    def __len__(self):
        """Return the total number of samples."""

        return len(self.image_paths)

    def __getitem__(self, idx):
        """Generate one sample of data."""

        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)

        Y = self.label_transform(label)
        X = self.image_transform(image)

        return X, Y
