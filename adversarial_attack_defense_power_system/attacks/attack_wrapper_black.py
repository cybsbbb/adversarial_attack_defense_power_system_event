import torch
from tqdm import tqdm
from adversarial_attack_defense_power_system.attacks.attacks_black import *


def attack_on_sample_black(sample_x, sample_y, net, attack_algorithm, attack_config):
    # Black-box Attacks
    # Score-Based attack
    if attack_algorithm == 'simba_attack':
        adv_x, success, query_cnt = simba_attack(net, sample_x, sample_y, attack_config)
    elif attack_algorithm == 'zo_sign_sgd_attack':
        adv_x, success, query_cnt = zo_sign_sgd_attack(net, sample_x, sample_y, attack_config)
    elif attack_algorithm == 'sign_hunter_attack':
        adv_x, success, query_cnt = sign_hunter_attack(net, sample_x, sample_y, attack_config)
    # Boundary-Based attack
    elif attack_algorithm == 'boundary_attack':
        adv_x, success, query_cnt = boundary_attack(net, sample_x, sample_y, attack_config)
    elif attack_algorithm == 'opt_attack':
        adv_x, success, query_cnt = opt_attack(net, sample_x, sample_y, attack_config)
    elif attack_algorithm == 'sign_opt_attack':
        adv_x, success, query_cnt = sign_opt_attack(net, sample_x, sample_y, attack_config)
    # Our Proposed Method
    elif attack_algorithm == 'bit_schedule_v4':
        adv_x, success, query_cnt = bit_schedule_attack_v4(net, sample_x, sample_y, attack_config)
    elif attack_algorithm == 'bit_schedule_v6':
        adv_x, success, query_cnt = bit_schedule_attack_v6(net, sample_x, sample_y, attack_config)
    elif attack_algorithm == 'bit_schedule_transfer_v1':
        adv_x, success, query_cnt = bit_schedule_attack_transfer_v1(net, sample_x, sample_y, attack_config)
    elif attack_algorithm == 'bit_schedule_transfer_v2':
        adv_x, success, query_cnt = bit_schedule_attack_transfer_v2(net, sample_x, sample_y, attack_config)
    else:
        print(f"Invalid attack type: {attack_algorithm}! Please re-check!")
        return None
    return adv_x, success, query_cnt


def attack_on_batch_black(batch_x, batch_y, net, attack_algorithm, attack_config):
    batch_x_adv = torch.zeros_like(batch_x)
    success_batch = []
    query_cnt_batch = []
    for idx in range(batch_x.shape[0]):
        # get sample
        sample_x = batch_x[idx:idx+1]
        sample_y = batch_y[idx:idx+1]
        # perform attack
        adv_x, success, query_cnt = attack_on_sample_black(sample_x, sample_y, net, attack_algorithm, attack_config)
        # set variables
        batch_x_adv[idx:idx+1] = adv_x
        success_batch.append(success)
        query_cnt_batch.append(query_cnt)
    return batch_x_adv, success_batch, query_cnt_batch


def attack_on_dataloader_black(data_loader, net, attack_algorithm, attack_config, device):
    net.eval()
    x_batches = []
    x_adv_batches = []
    success = []
    query_cnt = []
    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x_adv, success_batch, query_cnt_batch = attack_on_batch_black(batch_x, batch_y, net, attack_algorithm, attack_config)
        x_batches.append(batch_x)
        x_adv_batches.append(batch_x_adv)
        success += success_batch
        query_cnt += query_cnt_batch
    x = torch.cat(x_batches, axis=0)
    x_adv = torch.cat(x_adv_batches, axis=0)
    return x, x_adv, success, query_cnt
