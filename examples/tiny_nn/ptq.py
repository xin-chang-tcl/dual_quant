import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.quantization.quantize_fx import fuse_fx
from dual_quant.implementation.core.utils.model_utils import get_insert_fake_quant_model
from dual_quant.implementation.core.utils.training_utils import make_ptq_training
class RandomDataset(Dataset):
    def __init__(self, num_samples: int, input_size: tuple, num_classes: int = None):
        """
        Args:
            num_samples:    total number of samples in this fake dataset
            input_size:     shape of each sample, e.g. (3, 224, 224) or (10,)
            num_classes:    if provided, dataset will also return an int label in [0, num_classes)
        """
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # produce a single random sample
        x = torch.randn(self.input_size, dtype=torch.float)
        if self.num_classes is not None:
            y = torch.randint(0, self.num_classes, (1,)).item()
            return x, y
        return x

# Example usage:
if __name__ == "__main__":
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model = fuse_fx(model)
    input_shape = (5, 3, 56, 56)
    # 1,000 samples of 3×224×224 images, with 10 possible classes
    dataset = RandomDataset(num_samples=1000, input_size=(3, 56, 56))
    config = {

    }
    not_quantize=['features_0_0', 'features_1_conv_0_0']
    fake_quant_model = get_insert_fake_quant_model(model, torch.rand(input_shape),
                                                   per_tensor=False,
                                                   not_quantize_layers=not_quantize
                                                   )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    # Please provide input shape more than batch size 1, it will be used for per-channel case in initialization
    make_ptq_training(fake_quant_model, loader, input_shape=(input_shape), initial_iteration=10,
                      qsin_num_iteration=100, sequential_mode=False, lr=0.01)


