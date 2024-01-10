import torchvision.transforms.functional as TF
class Resizer():
    def __init__(self, size):
        self.size = size
    
    def __call__(self, sample):
        return TF.center_crop(sample, self.size)