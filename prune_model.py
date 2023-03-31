import os
import argparse
import torch
import torch.nn as nn
from models.mobilenet_pt import MobileNetv1
import torch_pruning as tp
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from set_seed import set_random_seed


# Argument parser
parser = argparse.ArgumentParser(description='EE361K project get latency')
parser.add_argument('--model', type=str,
                    default='mobilenetv1', help='Name of model')
parser.add_argument('--prune_ratio', type=float, help='pruning ratio')
parser.add_argument('--prune_metric', type=str,
                    default='l1', help='metric used for pruning')
parser.add_argument('--iter_steps', type=int, default=5,
                    help='the number of steps for pruning [only in Prune network mode]')
parser.add_argument('--finetune_epoch', type=int, default=5,
                    help='the number of epochs for finetuning')
args = parser.parse_args()


# Set random seed for reproducibility
set_random_seed(233)


model_str = args.model.lower()
if args.model.lower() == 'mobilenetv1':
    model = MobileNetv1()
os.makedirs('pruned', exist_ok=True)


ckpt = torch.load(f'ckpt/{model_str}.pt')
model.load_state_dict(ckpt)

# torch.save(model, f'pruned/{model_str}_raw.pth')

random_seed = 1
torch.manual_seed(random_seed)
batch_size = 128

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
criterion = criterion.to(torch.device('cuda:0'))
optimizer = torch.optim.Adam(model.parameters())


def fine_tune(model):
    model = model.train()
    device = torch.device('cuda:0')
    model = model.to(device)
    train_loss = 0
    train_total = 0
    train_correct = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # TODO: Put the images and labels on the GPU
        images = images.to(device)
        labels = labels.to(device)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()


def test(model):
    device = torch.device('cuda:0')
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    model = model.to(device)
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Perform the actual inference

            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test loss: %.4f Test accuracy: %.2f %%\n' %
          (test_loss / (batch_idx + 1), 100. * test_correct / test_total))


def prune_network(model, metric, prune_ratio, iterative_steps=5, fine_tune_epochs=1, ignored_layers=None):
    assert 0 < prune_ratio < 1
    model = model.to('cuda:0')
    example_inputs = torch.randn(1, 3, 32, 32).to(torch.device('cuda:0'))
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs)

    if metric.lower() == 'l1':
        imp = tp.importance.MagnitudeImportance(p=1)
    if metric.lower() == 'l2':
        imp = tp.importance.MagnitudeImportance(p=2)
    if ignored_layers is None:
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 10:
                ignored_layers.append(m)  # DO NOT prune the final classifier!

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=prune_ratio,
        ignored_layers=ignored_layers,
    )

    for i in range(iterative_steps):
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print('Ietration {} out of {}; Before fine-tuning'.format(i+1, iterative_steps))
        test(model)
        for j in range(fine_tune_epochs):
            fine_tune(model)
        print('Ietration {} out of {}; After fine-tuning'.format(i+1, iterative_steps))
        test(model)
        print(model)

    os.makedirs("./onnx_model", exist_ok=True)
    torch.onnx.export(model.to('cpu'),
                      torch.zeros((1, 3, 32, 32)),
                      'onnx_model/{}_{}_{}_{}_{}_pruned_net.onnx'.format(
                          model_str, metric, prune_ratio, iterative_steps, fine_tune_epochs),
                      input_names=['input'], opset_version=13)
    return model


print('Accuracy without pruning')
test(model)
print(model)
prune_network(model, args.prune_metric, args.prune_ratio,
              args.iter_steps, args.finetune_epoch)
