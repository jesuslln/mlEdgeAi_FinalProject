import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from models.mobilenet_pt import MobileNetv1


parser = argparse.ArgumentParser(description='EE361K final project')
parser.add_argument('--prune_ratio', type=float, help='pruning ratio', default=1.0)
args = parser.parse_args()


def prune_model_except_one_layer(model, layer_to_exclude, pruning_method, amount):
    for name, module in model.named_modules():
        if name != layer_to_exclude and isinstance(module, nn.Linear):
            parameters_to_prune = [(module, 'weight')]
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=pruning_method,
                amount=amount
            )


def find_layer_with_greatest_l1(model):
    max_l1 = 0
    layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            l1_norm = torch.norm(module.weight, p=1)
            if l1_norm > max_l1:
                max_l1 = l1_norm
                layer_name = name
    return layer_name


model = MobileNetv1()
layer_to_exclude = find_layer_with_greatest_l1(model) # I need to choose the layer with the highest L1 value
prune_ratio = args.prune_ratio # make args


prune_model_except_one_layer(
    model, layer_to_exclude, prune.L1Unstructured, prune_ratio)
