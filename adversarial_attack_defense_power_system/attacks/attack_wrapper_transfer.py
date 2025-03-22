from tqdm import tqdm
from adversarial_attack_defense_power_system.attacks.attacks_transfer import *


def attack_on_sample_transfer(sample_x, sample_y, net, surrogate_net, attack_algorithm, attack_config):
    if attack_algorithm == 'fgsm':
        adv_x, success = fgsm_attack_transfer(sample_x, sample_y, net, surrogate_net, attack_config)
    elif attack_algorithm == 'bim':
        adv_x, success = bim_attack_transfer(sample_x, sample_y, net, surrogate_net, attack_config)
    elif attack_algorithm == 'pgd':
        adv_x, success = pgd_attack_transfer(sample_x, sample_y, net, surrogate_net, attack_config)
    elif attack_algorithm == 'deepfool':
        adv_x, success = deepfool_attack_transfer(sample_x, sample_y, net, surrogate_net, attack_config)
    elif attack_algorithm == 'cw2':
        adv_x, success = carlini_wagner_l2_attack_transfer(sample_x, sample_y, net, surrogate_net, attack_config)
    else:
        print(f"Invalid attack type: {attack_algorithm}! Please re-check!")
        return None
    return adv_x, success


def attack_on_batch_transfer(batch_x, batch_y, net, surrogate_net, attack_algorithm, attack_config):
    batch_x_adv = torch.zeros_like(batch_x)
    success_batch = []
    for idx in range(batch_x.shape[0]):
        # get sample
        sample_x = batch_x[idx:idx+1]
        sample_y = batch_y[idx:idx+1]
        # perform attack
        adv_x, success = attack_on_sample_transfer(sample_x, sample_y, net, surrogate_net, attack_algorithm, attack_config)
        # set variables
        batch_x_adv[idx:idx+1] = adv_x
        success_batch.append(success)
    return batch_x_adv, success_batch


def attack_on_dataloader_transfer(data_loader, net, surrogate_net, attack_algorithm, attack_config, device):
    net.eval()
    surrogate_net.eval()
    x_batches = []
    x_adv_batches = []
    success = []
    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x_adv, success_batch = attack_on_batch_transfer(batch_x, batch_y, net, surrogate_net, attack_algorithm, attack_config)
        x_batches.append(batch_x)
        x_adv_batches.append(batch_x_adv)
        success += success_batch
    x = torch.cat(x_batches, axis=0)
    x_adv = torch.cat(x_adv_batches, axis=0)
    return x, x_adv, success
