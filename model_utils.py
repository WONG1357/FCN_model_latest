import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

def setup_fcn_model(num_classes, device):
    """Initialize the FCN8s model and move it to the specified device."""
    repo_parent_path = '/content/FCN-pytorch'
    if not os.path.exists(repo_parent_path):
        os.system('git clone -q https://github.com/pochih/FCN-pytorch.git')
    repo_code_path = '/content/FCN-pytorch/python'
    if repo_code_path not in sys.path:
        sys.path.append(repo_code_path)
    
    try:
        from fcn import FCN8s, VGGNet
        print("✅ Successfully imported FCN8s and VGGNet.")
    except ImportError as e:
        print(f"❌ FATAL: Could not import the model. Error: {e}")
        raise

    vgg_net = VGGNet(requires_grad=True)
    model = FCN8s(pretrained_net=vgg_net, n_class=num_classes)
    model.to(device)
    return model

def setup_optimizer(model, learning_rate):
    """Initialize the optimizer."""
    return optim.Adam(model.parameters(), lr=learning_rate)

def setup_criterion():
    """Initialize the loss function."""
    return nn.CrossEntropyLoss()