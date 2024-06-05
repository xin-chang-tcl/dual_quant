from dual_quant.implementation.core.utils.model_utils import get_insert_fake_quant_model
import torchvision
from torch.quantization.quantize_fx import fuse_fx
if __name__ == '__main__':
    config = {
        'per_channel': True,
        'lowest_scale': 1e-4,
        'weight_path': '/home/xin/devine/dual_quant/model_checkpoints/mobilenetv2.pth'
    }
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model = fuse_fx(model.eval())
    fake_quant_model = get_insert_fake_quant_model(model, (1, 3, 224, 224), config)