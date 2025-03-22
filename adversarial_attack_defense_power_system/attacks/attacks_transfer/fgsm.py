""" The FGSM transfer attack """
import torch
from adversarial_attack_defense_power_system.attacks.attacks_white.fgsm import fgsm_attack


def fgsm_attack_transfer(sample_x, sample_y, net, surrogate_net, attack_config=None):
    # Original Wrong Prediction, skip the attack.
    init_pred = net(sample_x).max(1, keepdim=True)[1]
    if init_pred.item() != sample_y.max(1, keepdim=True)[1].item():
        return sample_x, True

    # Use the surrogate_net to perform attack
    sample_x_adv, success = fgsm_attack(sample_x, sample_y, surrogate_net, attack_config=attack_config)

    # Adjust the perturbation norm
    epsilon_l2 = attack_config['epsilon_l2']
    perturbation = sample_x_adv - sample_x
    perturbation_norm = torch.linalg.vector_norm(torch.flatten(perturbation)).cpu().item()
    if perturbation_norm < 1e-5:
        return sample_x, False
    perturbation = perturbation * (epsilon_l2 / perturbation_norm)
    sample_x_adv = sample_x + perturbation

    # Check for success
    success = False
    if net(sample_x_adv).max(1, keepdim=True)[1].item() != sample_y.max(1, keepdim=True)[1].item():
        success = True

    return sample_x_adv, success
