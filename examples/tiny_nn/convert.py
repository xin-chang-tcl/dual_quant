import torch.quantization

from dual_quant.implementation.core.utils.model_utils import get_insert_fake_quant_model
import torchvision
from torch.quantization.quantize_fx import fuse_fx
if __name__ == '__main__':
    model = torchvision.models.mobilenet_v2(pretrained=True)
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
    fake_quant_model(torch.rand(input_shape))
    model.load_state_dict('ptq.pth or qat.pth')
    model.apply(torch.quantization.disable_observer)
    model_q = torch.quantization.convert(fake_quant_model.eval().cpu())
    converter = tinynn.converter.TFLiteConverter(model_q, torch.randn(2, 3, 56, 56), 'mv3sbn.tflite',
                                                 quantize_target_type='int8',
                                                 fuse_quant_dequant=True,
                                                 enable_mtk_ops=True,
                                                 rewrite_quantizable=True, )
    converter.convert()
    ### ready for training in original pipeline.
