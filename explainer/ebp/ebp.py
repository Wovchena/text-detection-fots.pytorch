import types
import torch
from explainer.ebp.functions import EBConv2d, EBLinear, EBAvgPool2d
from modules.parse_polys import parse_polys
import cv2
import numpy as np


def get_layer(model, key_list):
    a = model
    for key in key_list:
        a = a._modules[key]
    return a

class ExcitationBackpropExplainer(object):
    def __init__(self, model, output_layer_keys=None):
        self.output_layer = get_layer(model, output_layer_keys)
        self.model = model
        self._override_backward()
        self._register_hooks()

    def _override_backward(self):
        def new_linear(self, x):
            return EBLinear.apply(x, self.weight, self.bias)
        def new_conv2d(self, x):
            return EBConv2d.apply(x, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        def new_avgpool2d(self, x):
            return EBAvgPool2d.apply(x, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        def replace(m):
            name = m.__class__.__name__
            if name == 'Linear':
                m.forward = types.MethodType(new_linear, m)
            elif name == 'Conv2d':
                m.forward = types.MethodType(new_conv2d, m)
            elif name == 'AvgPool2d':
                m.forward = types.MethodType(new_avgpool2d, m)

        self.model.apply(replace)

    def _register_hooks(self):
        self.intermediate_vars = []
        def forward_hook(m, i, o):
            self.intermediate_vars.append(o)

        self.output_layer.register_forward_hook(forward_hook)

    def explain(self, inp):
        self.intermediate_vars = []

        raw_confidence, raw_distances, raw_angle = self.model(inp)
        confidence = raw_confidence.squeeze().data.cpu().numpy()
        distances = raw_distances.squeeze().data.cpu().numpy()
        angle = raw_angle.squeeze().data.cpu().numpy()
        polys = parse_polys(confidence, distances, angle, 0.95, 0.3).round().astype(int)
        output_var = self.intermediate_vars[0]

        scale_x = raw_confidence.shape[3] / inp.shape[3]
        scale_y = raw_confidence.shape[2] / inp.shape[2]

        excitations_per_poly = []
        for poly in polys:
            poly = poly[:8].reshape(4, 2)[np.newaxis, ...]
            grads = torch.zeros_like(raw_confidence).cpu().numpy()
            # grads[0, 0, poly[0, 1], poly[0, 0]] = 1
            # minAreaRect = cv2.minAreaRect(poly)
            # shrinkage = min(minAreaRect[1][0], minAreaRect[1][1]) * 0.6
            # shrunk_minAreaRect = minAreaRect[0], (minAreaRect[1][0] - shrinkage, minAreaRect[1][1] - shrinkage), \
            #                      minAreaRect[2]
            # shrunk_poly = cv2.boxPoints(shrunk_minAreaRect).round().astype(int)
            cv2.fillConvexPoly(grads[0, 0], (poly * np.array([scale_x, scale_y])).round().astype(int), 1)
            grads = torch.from_numpy(grads).to(raw_confidence.device)
            attmap_var = torch.autograd.grad(raw_confidence, output_var, grads, retain_graph=True)
            attmap = attmap_var[0].data.clone()
            attmap = torch.clamp(attmap.sum(1).unsqueeze(1), min=0.0)

            highlighted_img = attmap

            highlighted_img = highlighted_img.clamp(0.035 * highlighted_img.max(), highlighted_img.max())
            highlighted_img -= highlighted_img.min()
            highlighted_img /= highlighted_img.max()
            excitations_per_poly.append((highlighted_img, poly))
        return excitations_per_poly


class ContrastiveExcitationBackpropExplainer(object):
    def __init__(self, model, intermediate_layer_keys=None, output_layer_keys=None, final_linear_keys=None):
        self.intermediate_layer = get_layer(model, intermediate_layer_keys)
        self.output_layer = get_layer(model, output_layer_keys)
        self.final_linear = get_layer(model, final_linear_keys)
        self.model = model
        self._override_backward()
        self._register_hooks()

    def _override_backward(self):
        def new_linear(self, x):
            return EBLinear.apply(x, self.weight, self.bias)
        def new_conv2d(self, x):
            return EBConv2d.apply(x, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        def new_avgpool2d(self, x):
            return EBAvgPool2d.apply(x, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        def replace(m):
            name = m.__class__.__name__
            if name == 'Linear':
                m.forward = types.MethodType(new_linear, m)
            elif name == 'Conv2d':
                m.forward = types.MethodType(new_conv2d, m)
            elif name == 'AvgPool2d':
                m.forward = types.MethodType(new_avgpool2d, m)

        self.model.apply(replace)

    def _register_hooks(self):
        self.intermediate_vars = []
        def forward_hook(m, i, o):
            self.intermediate_vars.append(o)

        self.intermediate_layer.register_forward_hook(forward_hook)
        self.output_layer.register_forward_hook(forward_hook)

    def explain(self, inp, ind=None):
        self.intermediate_vars = []

        output = self.model(inp)
        output_var, intermediate_var = self.intermediate_vars

        if ind is None:
            ind = output.data.max(1)[1]
        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)

        self.final_linear.weight.data *= -1.0
        neg_map_var = torch.autograd.grad(output, intermediate_var, grad_out, retain_graph=True)
        neg_map = neg_map_var[0].data.clone()

        self.final_linear.weight.data *= -1.0
        pos_map_var = torch.autograd.grad(output, intermediate_var, grad_out, retain_graph=True)
        pos_map = pos_map_var[0].data.clone()

        diff = pos_map - neg_map
        attmap_var = torch.autograd.grad(intermediate_var, output_var, diff, retain_graph=True)

        attmap = attmap_var[0].data.clone()
        attmap = torch.clamp(attmap.sum(1).unsqueeze(1), min=0.0)

        return attmap
