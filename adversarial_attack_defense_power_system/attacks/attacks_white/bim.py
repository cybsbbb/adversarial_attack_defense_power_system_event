""" The BIM attack """
import torch
from adversarial_attack_defense_power_system.attacks.attacks_white.fgsm import get_data_grad


def bim_attack(sample_x, sample_y, net, attack_config=None):
    # Original Wrong Prediction, skip the attack.
    init_pred = net(sample_x).max(1, keepdim=True)[1]
    if init_pred.item() != sample_y.max(1, keepdim=True)[1].item():
        return sample_x, True
    # Get attack configs
    if attack_config is None:
        alpha = 0.005
        eps = 0.05
        max_iter = 20
    else:
        alpha = attack_config['alpha']
        eps = attack_config['eps']
        max_iter = attack_config['max_iter']
    # Perform attack
    r_tot = torch.zeros_like(sample_x)
    sample_x_adv = sample_x.clone().detach()
    loop_i = 0
    while net(sample_x).max(1, keepdim=True)[1].item() == sample_y.max(1, keepdim=True)[1].item() and loop_i < max_iter:
        # Get the gradient of the data
        data_grad = get_data_grad(sample_x_adv, sample_y, net)
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Aggregate the perturbation and clip
        r_tot += alpha * sign_data_grad
        r_tot = torch.clip(r_tot, min=-eps, max=eps)
        sample_x_adv = sample_x + r_tot
        loop_i += 1
    # Check for success
    success = False
    if net(sample_x_adv).max(1, keepdim=True)[1].item() != sample_y.max(1, keepdim=True)[1].item():
        success = True

    return sample_x_adv, success
