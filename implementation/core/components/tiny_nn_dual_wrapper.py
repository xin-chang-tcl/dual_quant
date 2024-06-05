from tinynn.graph.quantization.quantizer import QATQuantizer
import torch.nn as nn
import typing
import queue
import torch.nn.intrinsic as nni
from tinynn.util.train_util import get_logger, get_module_device
from tinynn.graph.tracer import TraceGraph
import sys
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, MinMaxObserver, PerChannelMinMaxObserver
from tinynn.graph.quantization import fused_modules as fm
from distutils.version import LooseVersion
from tinynn.graph.quantization.fake_quantize import FakeQuantizeTFLite
from dual_quant.implementation.core.components.dual_quantizer import DualQuantizer
from tinynn.graph.quantization.qat_modules import (
    Conv1d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTransposeBn2d,
)
import torch.nn.quantized.dynamic as nnqd
import copy
import torch.quantization as torch_q
import torch


log = get_logger(__name__, 'WARNING')

try:
    import ruamel_yaml as yaml
    from ruamel_yaml import CommentedMap
except ModuleNotFoundError:
    import ruamel.yaml as yaml
    from ruamel.yaml import CommentedMap
FUSE_QAT_MODULES = {
    nn.Conv1d: Conv1d,
    nn.ConvTranspose1d: ConvTranspose1d,
    nn.ConvTranspose2d: ConvTranspose2d,
    fm.ConvTransposeBn2d: ConvTransposeBn2d,
}

KNOWN_QSTATS = {
    nn.Softmax: (0, 256.0),
    'softmax': (0, 256.0),
    nn.LogSoftmax: (255, 16.0),
    'log_softmax': (255, 16.0),
}

FUSE_QAT_MODULES_CUSTOM = {}

if LooseVersion(torch.__version__) >= '1.13.0':
    from tinynn.graph.quantization.quantizable.gru import GRU

    FUSE_QAT_MODULES_CUSTOM.update({nn.GRU: GRU})


class PingPongwrapper(QATQuantizer):
    def __init__(self, model, dummy_input, work_dir=None, config=None, extra_param=None):
        super().__init__(model, dummy_input, work_dir, config)
        self.lowest_scale = config['lowest_scale']

    def prepare_qconfig(self, graph: TraceGraph, backend: str):
        """Prepare qconfig for various configurations.

        Args:
            graph (TraceGraph): The computation graph of the model
            backend (str, optional): The backend of quantization
        """

        log.info('setting qat backend and call prepare_qat')
        actual_backend = backend
        if backend in ('onnx', 'tensorrt'):
            actual_backend = 'qnnpack'
        # if not self.legacy_fq:
        #     qconfig = torch_q.get_default_qat_qconfig(actual_backend)
        # else:
        if LooseVersion(torch.__version__) >= '1.13.0':
            # See https://github.com/pytorch/pytorch/pull/88876
            qconfig = torch_q.QConfig(
                activation=DualQuantizer.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0, quant_max=255,
                    lowest_scale=self.lowest_scale,
                    dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                    reduce_range=False,
                ),
                weight=DualQuantizer.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=-127, quant_max=127,
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
            )
        else:
            version = None
            if LooseVersion(torch.__version__) >= '1.12.0':
                version = 0
            qconfig = torch_q.get_default_qat_qconfig(actual_backend, version)

        qconfig_c = None
        if self.rounding_mode == 'tflite':
            q_a = FakeQuantizeTFLite.with_args(*qconfig.activation.p.args, **qconfig.activation.p.keywords)
            q_w = FakeQuantizeTFLite.with_args(*qconfig.weight.p.args, **qconfig.weight.p.keywords)
            qconfig = torch_q.QConfig(q_a, q_w)
        if backend == 'qnnpack':
            if not self.asymmetric:
                sym_fq = qconfig.activation.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                )
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = qconfig.weight.with_args(
                    observer=PerChannelMinMaxObserver.with_args(quant_min=-128, quant_max=127),
                    quant_min=-127,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=False,
                    ch_axis=0,
                )
                qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        elif backend == 'fbgemm':
            fq_type = qconfig.weight.p.func
            sym_fq = fq_type.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False,
            )
            qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        elif backend in ('onnx', 'tensorrt'):
            if not self.asymmetric:
                sym_fq = qconfig.activation.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                )
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = qconfig.weight.with_args(
                    observer=MovingAveragePerChannelMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=False,
                    ch_axis=0,
                )
                qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        else:
            log.warning(f'Quantization backend {self.backend} is not tested. Please use at your risk.')

        torch.backends.quantized.engine = actual_backend
        graph.module.qconfig = qconfig
        if self.backend == 'qnnpack':
            if qconfig_c is not None:
                q = queue.Queue()
                q.put(graph.module)

                while not q.empty():
                    m = q.get()
                    if type(m).__name__ in (
                            'Conv2d',
                            'ConvBnReLU2d',
                            'ConvBn2d',
                            'ConvReLU2d',
                            'Conv1d',
                            'ConvBnReLU1d',
                            'ConvBn1d',
                    ):
                        m.qconfig = qconfig_c
                    else:
                        for c in m.children():
                            q.put(c)
        elif self.backend == 'fbgemm':
            if qconfig_c is not None:
                q = queue.Queue()
                q.put(graph.module)

                while not q.empty():
                    m = q.get()
                    if type(m).__name__ in ('Linear', 'LinearReLU'):
                        m.qconfig = qconfig_c
                    else:
                        for c in m.children():
                            q.put(c)

        def _lstm_node(node, custom_data):
            return isinstance(node.module, nn.LSTM)

        if self.dynamic_lstm_quant:
            lstm_nodes = graph.filter_forward_nodes(_lstm_node)
            for node in lstm_nodes:
                node.quantized = True
                node.module.qconfig = torch_q.default_dynamic_qconfig

    def prepare_qat(
        self,
        graph: TraceGraph,
        is_input_quantized: typing.Optional[typing.Tuple[bool]] = None,
        backend: str = 'qnnpack',
        fuse_only: bool = False,
    ) -> torch.nn.Module:
        """Prepare model for QAT training

        Args:
            graph (TraceGraph): The computation graph of the model
            is_input_quantized (typing.Union[typing.Tuple[bool]], optional): Whether the input tensor(s) is (are) \
                quantized. Defaults to None.
            backend (str, optional): The backend of quantization. Defaults to 'qnnpack'.
            fuse_only (bool, optional): Whether the returned model is only fused in PostQuantizer. Defaults to False.

        Returns:
            torch.nn.Module: The QAT-ready model
        """

        graph.module.train()

        self.prepare_qat_prep(graph, is_input_quantized, backend)

        if not graph.quantized:
            log.warning('Graph is not quantized, skip preparation')
            return graph.module

        # Unfornately, the suggested way below will try to fuse all the modules
        # even if some of the nodes are not in a quantized computation graph.
        # So we wrote some alternatives for the function.
        #   torch.quantization.prepare_qat(graph.module, inplace=True)

        if hasattr(torch_q, 'get_default_qat_module_mappings'):
            mapping = torch_q.get_default_qat_module_mappings()
        elif hasattr(torch_q, 'get_qat_module_mappings'):
            mapping = copy.deepcopy(torch_q.get_qat_module_mappings())
        else:
            mapping = copy.deepcopy(torch_q.DEFAULT_QAT_MODULE_MAPPING)

        if self.dynamic_lstm_quant:
            mapping = dict(mapping)
            mapping.update({nn.LSTM: nnqd.LSTM})

        mapping.update(FUSE_QAT_MODULES)

        q_dict = {}
        for n in graph.forward_nodes:
            if isinstance(n.module, nn.Module):
                q_dict[n.module] = q_dict.get(n.module, False) or n.quantized

        non_quantized_mods = set(m for m, s in q_dict.items() if s is False)

        if LooseVersion(torch.__version__) < LooseVersion("1.7.0"):
            model = torch_q.prepare(graph.module, inplace=True)
            for m in non_quantized_mods:
                if hasattr(m, "_forward_hooks"):
                    if len(m._forward_hooks) > 0:
                        m._forward_hooks.popitem()
                if hasattr(m, "qconfig"):
                    delattr(m, "qconfig")
                if hasattr(m, "activation_post_process"):
                    delattr(m, "activation_post_process")

            if hasattr(graph.module, 'activation_post_process'):
                if hasattr(graph.module, "_forward_hooks"):
                    if len(graph.module._forward_hooks) > 0:
                        graph.module._forward_hooks.popitem()
                delattr(graph.module, "activation_post_process")

            torch_q.convert(model, mapping, inplace=True)
        else:
            torch_q.propagate_qconfig_(graph.module, qconfig_dict=None)
            for m in non_quantized_mods:
                if hasattr(m, "qconfig"):
                    delattr(m, "qconfig")
            model = torch_q.convert(graph.module, mapping=mapping, inplace=True, remove_qconfig=False)

            if self.dynamic_lstm_quant:
                mapping.pop(nn.LSTM)

            if LooseVersion(torch.__version__) >= LooseVersion("1.13.0"):
                torch_q.propagate_qconfig_(model, qconfig_dict=None)

                for m in non_quantized_mods:
                    if hasattr(m, "qconfig"):
                        delattr(m, "qconfig")

                prepare_custom_config_dict = torch.ao.quantization.get_default_custom_config_dict()
                custom_module_class_mapping = prepare_custom_config_dict.get(
                    "float_to_observed_custom_module_class", {}
                )
                custom_module_class_mapping.update(FUSE_QAT_MODULES_CUSTOM)
                qconfig_propagation_list = torch_q.get_default_qconfig_propagation_list()

                from tinynn.graph.quantization.quantizable import lstm, gru

                orig_from_float = torch.ao.nn.quantizable.LSTM.from_float

                torch.ao.nn.quantizable.LSTM.from_float = lstm.from_float
                GRU.from_float = gru.from_float

                def patch_observer_set(orig_func):
                    def new_no_observer_set():
                        return set(FUSE_QAT_MODULES_CUSTOM.values()) | orig_func()

                    return new_no_observer_set

                orig_no_observer_set = sys.modules['torch.ao.quantization.quantize'].no_observer_set
                sys.modules['torch.ao.quantization.quantize'].no_observer_set = patch_observer_set(orig_no_observer_set)

                if hasattr(torch_q, 'add_observer_'):
                    add_observer_func = torch_q.add_observer_
                else:
                    #add_observer_func = _add_observer_
                    add_observer_func = sys.modules['torch.ao.quantization.quantize']._add_observer_

                add_observer_func(
                    model,
                    qconfig_propagation_list,
                    set(mapping.values()),
                    custom_module_class_mapping=custom_module_class_mapping,
                )

                torch.ao.nn.quantizable.LSTM.from_float = orig_from_float
                sys.modules['torch.ao.quantization.quantize'].no_observer_set = orig_no_observer_set

            else:
                torch_q.prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)
            for m in non_quantized_mods:
                if hasattr(m, "_forward_hooks"):
                    if len(m._forward_hooks) > 0:
                        m._forward_hooks.popitem()
                if hasattr(m, "qconfig"):
                    delattr(m, "qconfig")
                if hasattr(m, "activation_post_process"):
                    delattr(m, "activation_post_process")

        if not self.per_tensor:
            if self.backend == 'qnnpack':
                for n, m in graph.module.named_modules():
                    if n.endswith('.weight_fake_quant'):
                        observer = getattr(m, 'activation_post_process', None)
                        if observer is not None:
                            m.quant_min = -127
                            m.quant_max = 127
                            observer.quant_min = -127
                            observer.quant_max = 127

        self.extra_qat_fusion_postprocess(graph)

        if self.disable_requantization_for_cat:
            self.disable_requantization_for_cat_pass(graph)

        if self.quantized_input_stats is not None:
            self.prepare_quantized_inputs_pass(graph)

        if self.set_quantizable_op_stats:
            if self.quantized_op_stats is None:
                self.quantized_op_stats = {}
            self.quantized_op_stats.update(KNOWN_QSTATS)

        if self.quantized_op_stats is not None:
            self.prepare_quantized_ops_pass(graph)

        if self.backend == 'tensorrt':
            q = queue.Queue()
            q.put(('', graph.module))

            while not q.empty():
                s, m = q.get()
                if isinstance(m, torch_q.QuantStub):
                    m.apply(torch.quantization.enable_fake_quant)
                    m.apply(torch.quantization.enable_observer)

                elif hasattr(m, 'activation_post_process'):
                    m.activation_post_process.apply(torch.quantization.disable_fake_quant)
                    m.activation_post_process.apply(torch.quantization.disable_observer)

                    if hasattr(m, 'weight_fake_quant'):
                        m.weight_fake_quant.apply(torch.quantization.enable_fake_quant)
                        m.weight_fake_quant.apply(torch.quantization.enable_observer)
                else:
                    for n, c in m.named_children():
                        q.put((f'{s}{n}', c))

        if self.backend == 'onnx':
            self.leaf_nodes = []
            self.swap_nodes = []

            q = queue.Queue()
            for node in graph.output_nodes:
                for prev_node in node.prev_nodes:
                    q.put((prev_node, False))

            while not q.empty():
                n, state = q.get()
                if isinstance(n.module, nn.Module):
                    orig_name = graph.module_original_name_dict.get(id(n.module))
                    new_mod, _ = graph.get_submodule_with_parent_from_name(orig_name, self.inplace)
                    if isinstance(new_mod, torch_q.DeQuantStub):
                        state = True
                    else:
                        if state:
                            if isinstance(new_mod, torch_q.QuantStub):
                                state = False
                            elif isinstance(new_mod, nn.Module) and hasattr(new_mod, 'activation_post_process'):
                                self.leaf_nodes.append(new_mod)
                                state = False
                            elif (
                                isinstance(new_mod, nn.Sequential)
                                and type(new_mod).__module__.startswith(nni.__name__)
                                and len(new_mod) > 0
                                and hasattr(new_mod[-1], 'activation_post_process')
                            ):
                                self.leaf_nodes.append(new_mod[-1])
                                state = False
                    for pn in n.prev_nodes:
                        q.put((pn, state))

            q = queue.Queue()
            visited = set()
            for node in graph.input_nodes:
                q.put((node, None, False, 0))

            while not q.empty():
                n, prev_q_mod, state, idx = q.get()
                key = f'{n.unique_name}:{idx}'
                if key in visited:
                    continue
                else:
                    visited.add(key)

                q_mod = prev_q_mod
                if n.quantized:
                    if isinstance(n.module, nn.Module):
                        orig_name = graph.module_original_name_dict.get(id(n.module))
                        new_mod, _ = graph.get_submodule_with_parent_from_name(orig_name, self.inplace)
                        if isinstance(new_mod, nn.Module) and hasattr(new_mod, 'activation_post_process'):
                            q_mod = new_mod
                        elif (
                            isinstance(new_mod, nn.Sequential)
                            and type(new_mod).__module__.startswith(nni.__name__)
                            and len(new_mod) > 0
                            and hasattr(new_mod[-1], 'activation_post_process')
                        ):
                            q_mod = new_mod[-1]
                        elif isinstance(new_mod, torch_q.DeQuantStub):
                            q_mod = new_mod
                        elif type(new_mod) != nn.Identity:
                            state = True
                    else:
                        is_prev_float_functional = (
                            len(n.prev_nodes) > 1 and n.prev_nodes[0].type() == torch.nn.quantized.FloatFunctional
                        )
                        if is_prev_float_functional:
                            q_mod = getattr(n.prev_nodes[0].module, n.kind())
                        else:
                            state = True

                    if state and prev_q_mod is not None and q_mod != prev_q_mod:
                        self.swap_nodes.append((prev_q_mod, q_mod, idx))
                        state = False

                for next_n in n.next_nodes:
                    idx = next_n.prev_nodes.index(n)
                    q.put((next_n, q_mod, state, idx))

        return graph.module