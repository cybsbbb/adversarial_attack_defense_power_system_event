import torch.nn.functional as F


def fgsm_attack(sample_x, sample_y, net, attack_config=None):
    # Original Wrong Prediction, skip the attack.
    init_pred = net(sample_x).max(1, keepdim=True)[1]
    if init_pred.item() != sample_y.max(1, keepdim=True)[1].item():
        return sample_x, True
    # Get attack configs
    if attack_config is None:
        epsilon = 0.05
    else:
        epsilon = attack_config['epsilon']
    # Get the gradient of the data
    data_grad = get_data_grad(sample_x, sample_y, net)
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed sample
    sample_x_adv = sample_x + epsilon * sign_data_grad

    # Check for success
    success = False
    if net(sample_x_adv).max(1, keepdim=True)[1].item() != sample_y.max(1, keepdim=True)[1].item():
        success = True
    return sample_x_adv, success


def get_data_grad(sample_x, sample_y, net):
    # Set requires_grad attribute of tensor. Important for Attack
    sample_x.requires_grad = True
    # print(sample_x.requires_grad)
    # Forward pass the data through the model
    output = net(sample_x)
    # Calculate the loss
    loss = F.cross_entropy(output, sample_y)
    # Zero all existing gradients
    net.zero_grad()
    # Calculate gradients of model in backward pass
    loss.backward()
    # Collect ``datagrad``
    data_grad = sample_x.grad.data
    return data_grad
