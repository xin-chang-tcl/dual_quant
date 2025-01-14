from tinynn.graph.quantization.quantizer import QATQuantizer
import torch.nn as nn
import typing
import queue
import re
import torch.nn.intrinsic as nni
from tinynn.util.train_util import get_logger, get_module_device
from tinynn.graph.tracer import TraceGraph
import sys
import torch
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver
from tinynn.graph.quantization import fused_modules as fm
from distutils.version import LooseVersion
import torch.nn.quantized as nnq
from tinynn.graph.quantization.fake_quantize import FakeQuantizeTFLite
from tinynn.graph.quantization.quantizer import PostQuantizer, TraceNode, TraceFunction, load_creation_func_names, load_processed_ptq_rules, load_processed_qat_rules, Q_MODULES_MAPPING
from dual_quant.implementation.core.components.dual_quantizer import DualQuantizer
from tinynn.graph.quantization.qat_modules import (
    Conv1d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTransposeBn2d,
)
import functools
import torch.nn.quantized.dynamic as nnqd
import copy
import torch.quantization as torch_q
import torch

from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_static_quant_module_mappings,
    get_default_static_quant_reference_module_mappings,
    get_default_qat_module_mappings,
    get_default_qconfig_propagation_list,
    no_observer_set,
    _has_special_act_post_process,
    _get_special_act_post_process,
)
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


class Dualwrapper(QATQuantizer):
    def __init__(self, model, dummy_input, work_dir=None, config=None, extra_param=None):
        super().__init__(model, dummy_input, work_dir, config)
        self.lowest_scale = config['lowest_scale']
        self.threshold = config['threshold']
        self.penalty_factor = config['penalty_factor']
        self.fuse_bn = config['fuse_bn']

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
                    threshold=self.threshold,
                    penalty_factor=self.penalty_factor,
                    dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                    reduce_range=False,
                ),
                weight=DualQuantizer.with_args(
                    observer=MinMaxObserver,
                    quant_min=-127, quant_max=127,
                    lowest_scale=self.lowest_scale,
                    threshold=self.threshold,
                    penalty_factor=self.penalty_factor,
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
                    quant_min=0,
                    quant_max=255,
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                )
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = qconfig.weight.with_args(
                    observer=torch_q.PerChannelMinMaxObserver.with_args(quant_min=-127, quant_max=127),
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
        from torch.ao.quantization.quantization_mappings import DEFAULT_MODULE_TO_ACT_POST_PROCESS

        # if nn.Sigmoid in DEFAULT_MODULE_TO_ACT_POST_PROCESS:
        #     DEFAULT_MODULE_TO_ACT_POST_PROCESS[nn.Sigmoid] = qconfig.activation

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
                print('added script!')
                if hasattr(torch_q, 'add_observer_'):
                    add_observer_func = torch_q.add_observer_
                else:
                    add_observer_func = sys.modules['torch.ao.quantization.quantize']._add_observer_
                #add_observer_func = _add_observer_

                    #add_observer_func = sys.modules['torch.ao.quantization.quantize']._add_observer_
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


    def prepare_qat_prep(
        self,
        graph: TraceGraph,
        is_input_quantized: typing.Optional[typing.Tuple[bool]] = None,
        backend: str = 'qnnpack',
    ):
        """Some common logic before calling torch.quantization.prepare[_qat]

        Args:
            graph (TraceGraph): The computation graph of the model
            is_input_quantized (typing.Union[typing.Tuple[bool]], optional): Whether the input tensor(s) is (are) \
                quantized. Defaults to None.
            backend (str, optional): The backend of quantization. Defaults to 'qnnpack'.

        """

        qat_analysis_queue = queue.Queue()
        visited = set()

        def _qat_analysis(node: TraceNode, quantized: bool):
            # Find quantized subgraphs in the whole computation graph

            if node.unique_name in visited:
                return

            visited.add(node.unique_name)

            if node in graph.output_nodes:
                return

            # TODO: Enable QAT analysis for TensorRT
            if self.backend == 'tensorrt':
                quantized = True
                node.quantized = True
            elif type(node.module) is torch_q.QuantStub:
                quantized = True
                node.quantized = quantized
            elif type(node.module) is torch_q.DeQuantStub:
                node.quantized = True
                quantized = False
            else:
                node.quantized = quantized
                log.debug(f"[QUANTIZED]{node.unique_name}:{quantized}")

            for i, n in enumerate(node.next_nodes):
                if type(n.module) == TraceFunction:
                    if n.kind() in ('shape', 'size', 'dtype', 'device'):
                        continue
                    if n.kind() == 'expand_as' and i > 0:
                        continue
                qat_analysis_queue.put((n, quantized))

        if is_input_quantized is not None:
            assert len(is_input_quantized) == len(graph.input_nodes)

            for n, q in zip(graph.input_nodes, is_input_quantized):
                qat_analysis_queue.put((n, q))
        else:
            for n in graph.input_nodes:
                qat_analysis_queue.put((n, False))

        creation_func_names = load_creation_func_names()

        def _is_extra_constant_nodes(node, custom_data):
            return node.full_name() in creation_func_names

        extra_constant_nodes = graph.filter_forward_nodes(_is_extra_constant_nodes)

        def _is_params_in_module(node, custom_data):
            if len(node.prev_nodes) == 1 and len(node.next_nodes) == 1:
                if len(node.prev_tensors) == 1 and len(node.next_tensors) == 1:
                    if isinstance(node.prev_tensors[0], nn.Module) and not isinstance(
                        node.prev_tensors[0], nnq.FloatFunctional
                    ):
                        return True
            return False

        param_nodes = graph.filter_forward_nodes(_is_params_in_module)

        for n in graph.constant_nodes + extra_constant_nodes:
            qat_analysis_queue.put((n, not torch.is_floating_point(n.next_tensors[0])))

        while not qat_analysis_queue.empty():
            node, quantized = qat_analysis_queue.get()
            if not graph.quantized:
                graph.quantized = graph.quantized or quantized
            _qat_analysis(node, quantized)

        q_dict = {}
        for n in graph.forward_nodes:
            if isinstance(n.module, nn.Module):
                q_dict[n.module] = q_dict.get(n.module, False) or n.quantized

        for n in graph.forward_nodes:
            if n.module in q_dict:
                n.quantized = q_dict[n.module]

        for n in param_nodes:
            prev_node = n.prev_nodes[0]
            next_node = n.next_nodes[0]
            is_known_mod = prev_node.kind().__name__ in (
                'Conv1d',
                'Conv2d',
                'Linear',
                'ConvTranspose1d',
                'ConvTranspose2d',
            )
            if is_known_mod and n.module.full_name == 'weight' and prev_node.quantized:
                if next_node.type() == torch_q.QuantStub:
                    mod = nn.Sequential(torch_q.DeQuantStub(), torch_q.QuantStub())
                    orig_mod = next_node.module
                    next_node.module = mod
                    setattr(graph.module, next_node.original_name, next_node.module)
                    graph.module_original_name_dict[id(mod)] = graph.module_original_name_dict[id(orig_mod)]
                    graph.module_unique_name_dict[id(mod)] = graph.module_unique_name_dict[id(orig_mod)]
                continue
            qat_analysis_queue.put((n, not torch.is_floating_point(n.next_tensors[0])))

        while not qat_analysis_queue.empty():
            node, quantized = qat_analysis_queue.get()
            if not graph.quantized:
                graph.quantized = graph.quantized or quantized
            _qat_analysis(node, quantized)

        log.debug("qat analysis over")

        if not graph.quantized:
            return

        if isinstance(self, PostQuantizer):
            processed_rules = load_processed_ptq_rules()
        else:
            processed_rules = load_processed_qat_rules()

        is_fusable = functools.partial(self.is_fusable, current_rules=processed_rules, graph=graph)

        def _find_quantized_module_nodes(node: TraceNode, custom_node):
            # Find quantized module nodes
            return node.type() in Q_MODULES_MAPPING and node.quantized

        # Replace module nodes with our custom variants
        quantized_mod_nodes = graph.filter_forward_nodes(_find_quantized_module_nodes)

        type_dict = {}
        for node in quantized_mod_nodes:
            node_type = node.type()
            type_dict.setdefault(node_type, [])
            type_dict[node_type].append(node)

        for node_type, nodes in type_dict.items():
            graph.update_submodule_in_nodes_from_predicate(nodes, Q_MODULES_MAPPING[node_type], self.inplace)

        custom_data = ([], set())
        graph.filter_forward_nodes(is_fusable, custom_data, reverse=True)
        quant_list = custom_data[0]
        log.info(f'found nodes to fuse: {quant_list}')

        new_fuser_func = fm.gen_fuse_known_modules_wrapper(
            sys.modules['torch.quantization.fuse_modules'].fuse_known_modules
        )
        if self.fuse_bn:
            if self.backend != 'tensorrt':
                is_qat = type(self) is QATQuantizer
                for quant_nodes in quant_list:
                    if self.inplace:
                        quant_nodes = [re.sub('get_submodule\\("(.*?)"\\)', '\\1', x) for x in quant_nodes]
                        quant_nodes = [re.sub('\\[("|)(.*?)("|)\\]', '.\\2', x) for x in quant_nodes]

                    if LooseVersion(torch.__version__) >= '1.11.0' and LooseVersion(torch.__version__) < '1.14.0':
                        # See https://github.com/pytorch/pytorch/pull/88193
                        sys.modules['torch.quantization.fuse_modules']._fuse_modules(
                            graph.module, quant_nodes, is_qat=is_qat, inplace=True, fuser_func=new_fuser_func
                        )
                    elif is_qat and LooseVersion(torch.__version__) >= '1.14.0':
                        # See https://github.com/pytorch/pytorch/issues/74028
                        torch.ao.quantization.fuse_modules_qat(
                            graph.module, quant_nodes, fuser_func=new_fuser_func, inplace=True
                        )
                    else:
                        torch_q.fuse_modules(graph.module, quant_nodes, fuser_func=new_fuser_func, inplace=True)

        self.prepare_qconfig(graph, backend)
        self.override_qconfig(graph.module)

# def _add_observer_(module, qconfig_propagation_list=None, non_leaf_module_list=None, device=None, custom_module_class_mapping=None):
#     r"""Add observer for the leaf child of the module.
#
#     This function insert observer module to all leaf child module that
#     has a valid qconfig attribute.
#
#     Args:
#         module: input module with qconfig attributes for all the leaf modules that we want to quantize
#         qconfig_propagation_list: a list of quantizable modules that will have observers added to them
#             if they are leaf nodes
#         device: parent device, if any
#         non_leaf_module_list: list of non-leaf modules we want to add observer
#
#     Return:
#         None, module is modified inplace with added observer modules and forward_hooks
#     """
#     if qconfig_propagation_list is None:
#         qconfig_propagation_list = get_default_qconfig_propagation_list()
#
#     if custom_module_class_mapping is None:
#         custom_module_class_mapping = {}
#
#     # respect device affinity when adding observers
#     def _get_unique_devices_(module):
#         return {p.device for p in module.parameters()} | \
#             {p.device for p in module.buffers()}
#     if device is None:
#         devices = _get_unique_devices_(module)
#         assert len(devices) <= 1, (
#             f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
#         )
#         device = next(iter(devices)) if len(devices) > 0 else None
#
#     def get_activation_post_process(qconfig, device, special_act_post_process=None):
#         activation = qconfig.activation() if special_act_post_process is None else special_act_post_process()
#         if device is not None:
#             activation.to(device)
#         return activation
#
#     def needs_observation(m):
#         return hasattr(m, 'qconfig') and m.qconfig is not None
#
#     def insert_activation_post_process(m, special_act_post_process=None):
#         """ Adds an activation post process module and register
#         a pre or post hook that calls the module
#         """
#         # We don't insert observer/fake_quantize for DeQuantStub
#         from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper
#         def _observer_forward_hook(self, input, output):
#             r"""Forward hook that calls observer on the output
#             """
#             return self.activation_post_process(output)
#
#         def _observer_forward_pre_hook(self, input):
#             r"""Forward pre hook that calls observer on the output
#             """
#             return self.activation_post_process(input[0])
#         from torch.quantization.qconfig import QConfig
#         from torch.quantization.fake_quantize import FakeQuantizeBase
#         def _activation_is_memoryless(qconfig: QConfig):
#             """
#             Return whether the observer for activations defined in the given QConfig is memoryless.
#             This means a MovingAverage observer with averaging constant equal to 1.
#             """
#
#             def _is_memoryless(observer):
#                 return hasattr(observer, "averaging_constant") and observer.averaging_constant == 1
#
#             act = qconfig.activation()
#             if isinstance(act, FakeQuantizeBase) and hasattr(act, "activation_post_process"):
#                 return _is_memoryless(act.activation_post_process)
#             else:
#                 return _is_memoryless(act)
#
#         def _register_activation_post_process_hook(module, pre_hook=False):
#             assert hasattr(module, 'activation_post_process'), \
#                 'Expect activation_post_process attribute already attached to the module'
#             if pre_hook:
#                 handle = module.register_forward_pre_hook(
#                     _observer_forward_pre_hook#, prepend=True
#                 )
#             else:
#                 handle = module.register_forward_hook(
#                     _observer_forward_hook#, prepend=True
#                 )
#         if needs_observation(m) and not isinstance(m, DeQuantStub):
#             # observer and hook will be gone after we swap the module
#             m.add_module('activation_post_process', get_activation_post_process(
#                 m.qconfig, device, special_act_post_process))
#             # Register observer as the first entry in the hook list
#             # All post forward hooks are preserved and will be executed after the observer before convert
#             _register_activation_post_process_hook(m, pre_hook=_activation_is_memoryless(m.qconfig))
#
#     from torch.nn.utils.parametrize import type_before_parametrizations
#     import torch.ao.nn.quantized as nnq
#     from torch.ao.nn.intrinsic import _FusedModule
#     for name, child in module.named_children():
#         # TODO remove Dropout special after codebase stable
#         if type_before_parametrizations(child) in [nn.Dropout]:
#             continue
#         elif issubclass(type_before_parametrizations(child), (nnq.FloatFunctional, nnq.QFunctional)):
#             if needs_observation(child):
#                 assert hasattr(child, "activation_post_process"), (
#                     f"functional class {type_before_parametrizations(child)} has no pre-defined `activation_post_process`"
#                 )
#                 child.activation_post_process = get_activation_post_process(child.qconfig, device)
#         elif isinstance(child, _FusedModule):
#             # activation_post_process are now added directly to nn.Sequential/_FusedModule
#             if needs_observation(child):
#                 insert_activation_post_process(child)
#         elif non_leaf_module_list is not None and type_before_parametrizations(child) in non_leaf_module_list:
#             if needs_observation(child):
#                 insert_activation_post_process(child)
#         elif _has_special_act_post_process(child):
#             #special_act_post_process = _get_special_act_post_process(child)
#             #insert_activation_post_process(child, special_act_post_process)
#             insert_activation_post_process(child)
#         elif needs_observation(child) and type_before_parametrizations(child) in custom_module_class_mapping:
#             observed_child = custom_module_class_mapping[type_before_parametrizations(child)].from_float(child)
#             setattr(module, name, observed_child)
#             # TODO: These are the modules that cannot be observed
#             #       Once there are more, we should move them to a separate list
#             if custom_module_class_mapping[type_before_parametrizations(child)] not in no_observer_set():
#                 insert_activation_post_process(observed_child)
#         else:
#             _add_observer_(child, qconfig_propagation_list, non_leaf_module_list, device, custom_module_class_mapping)
#
#     # Insert observers only for leaf nodes, note that this observer is for
#     # the output of the module, for input QuantStub will observe them
#     from torch.ao.quantization.utils import get_qparam_dict, has_no_children_ignoring_parametrizations
#     if has_no_children_ignoring_parametrizations(module) and not isinstance(module, torch.nn.Sequential) \
#        and type_before_parametrizations(module) in qconfig_propagation_list:
#         insert_activation_post_process(module)
