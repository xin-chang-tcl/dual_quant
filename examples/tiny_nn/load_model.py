from pingpong_quant.implementation.core.utils.model_utils import get_insert_fake_quant_model, convert_model_to_tflite
import torchvision
from torch.quantization.quantize_fx import fuse_fx
from pingpong_quant.implementation.core.components.pingpong_quantizer import prepare_for_convert
if __name__ == '__main__':
    config = {
        'per_channel': True,
        'lowest_scale': 1e-4,
        'weight_path': '/nas/projects/auto-ml/models/xin/prepare.result.enetb0.nosche.bs32.400/best_model.pth'
    }
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model = fuse_fx(model.eval())
    fake_quant_model = get_insert_fake_quant_model(model, (1, 3, 224, 224), config)
    fake_quant_model.apply(prepare_for_convert)