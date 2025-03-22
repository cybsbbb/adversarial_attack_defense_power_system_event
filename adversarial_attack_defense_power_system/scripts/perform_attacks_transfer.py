import os
import time
import csv
import json
import numpy as np
import torch
from adversarial_attack_defense_power_system.classifiers.load_classifier import load_classifier
from adversarial_attack_defense_power_system.dataset_loader.pmu_event_dataset import PMUEventDataset
from adversarial_attack_defense_power_system.attacks.attack_wrapper_transfer import attack_on_dataloader_transfer
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds
from pathlib import Path


def perform_attack_transfer(interconnection, model_name,
                            surrogate_interconnection, surrogate_model_name,
                            attack_algorithm, attack_config, device):
    setup_random_seeds(428)
    # Setup save path
    script_dir = Path(__file__).resolve().parent
    result_dir = (f"{script_dir}/../../adv_exp_result/transfer/"
                  f"{interconnection}_{model_name}_by_{surrogate_interconnection}_{surrogate_model_name}/"
                  f"{attack_algorithm}")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    exp_name = f"{interconnection}_{model_name}_by_{surrogate_interconnection}_{surrogate_model_name}_{attack_algorithm}"
    result_sub_dir = f"{result_dir}/{exp_name}"
    print(result_sub_dir)
    if not os.path.exists(result_sub_dir):
        os.makedirs(result_sub_dir)

    # Load the dataset
    testset = PMUEventDataset(interconnection=interconnection, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Load the trained classifier
    net = load_classifier(interconnection=interconnection, model_name=model_name, device=device)

    # Load the trained surrogate classifier
    surrogate_net = load_classifier(interconnection=surrogate_interconnection, model_name=surrogate_model_name, device=device)

    # Perform attack
    start_time = time.time()
    x, x_adv, success = attack_on_dataloader_transfer(testloader, net, surrogate_net, attack_algorithm, attack_config, device)
    end_time = time.time()

    # Stat the attack success rate and average query
    n = len(success)
    attack_success_rate = sum(map(int, success)) / n

    # Save result
    elapsed_time = end_time - start_time
    information = {"attack_algorithm": attack_algorithm,
                   "attack_config": attack_config,
                   "elapsed_time": elapsed_time,
                   "attack_success_rate": attack_success_rate,
                   "success": success,
                   }
    np.save(f'{result_sub_dir}/x_adv.npy', x_adv.detach().cpu().numpy())
    np.save(f'{result_sub_dir}/x.npy', x.detach().cpu().numpy())
    with open(f'{result_sub_dir}/attack_res.json', 'w') as fp:
        json.dump(information, fp, indent=4)
    print(f"exp_name: {exp_name}, attack_success_rate: {attack_success_rate}")
    return attack_success_rate


def transfer_experiments(name, target_model_list, surrogate_model_list, attack_algorithm, attack_config, device):
    script_dir = Path(__file__).resolve().parent
    result_dir = (f"{script_dir}/../../adv_exp_result/transfer/")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(f'{result_dir}/{name}_result_table.csv', 'w', newline='') as csvfile:
        fieldnames = ['target_model']
        for surrogate_interconnection, surrogate_model_name in surrogate_model_list:
            fieldnames.append(f'{surrogate_interconnection}_{surrogate_model_name}')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for interconnection, model_name in target_model_list:
            row = {'target_model': f'{interconnection}_{model_name}'}
            for surrogate_interconnection, surrogate_model_name in surrogate_model_list:
                attack_success_rate = perform_attack_transfer(interconnection, model_name,
                                                              surrogate_interconnection, surrogate_model_name,
                                                              attack_algorithm, attack_config, device)
                row[f'{surrogate_interconnection}_{surrogate_model_name}'] = attack_success_rate
            writer.writerow(row)
    return


def comprehensive_experiments(target_model_list, surrogate_model_list, attack_algorithm, attack_config, device):
    script_dir = Path(__file__).resolve().parent
    result_dir = (f"{script_dir}/../../adv_exp_result/transfer/{attack_algorithm}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(f'{result_dir}/result_table.csv', 'w', newline='') as csvfile:
        fieldnames = ['target_model']
        for surrogate_interconnection, surrogate_model_name in surrogate_model_list:
            fieldnames.append(f'{surrogate_interconnection}_{surrogate_model_name}')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for interconnection, model_name in target_model_list:
            row = {'target_model': f'{interconnection}_{model_name}'}
            for surrogate_interconnection, surrogate_model_name in surrogate_model_list:
                attack_success_rate = perform_attack_transfer(interconnection, model_name,
                                                              surrogate_interconnection, surrogate_model_name,
                                                              attack_algorithm, attack_config, device)
                row[f'{surrogate_interconnection}_{surrogate_model_name}'] = attack_success_rate
            writer.writerow(row)


if __name__ == '__main__':
    # Set up the device
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")

    interconnection_list = ['b', 'c']
    # model_name_list = ['vgg13', 'mobilenet_v2', 'efficientnet', 'densenet121', 'resnet18', 'resnet50']
    model_name_list = ['vgg13', 'mobilenet_v2', 'densenet121', 'resnet18', 'resnet50']
    attack_info_list = [{'attack_algorithm': 'fgsm', 'attack_config': {'epsilon': 0.1, 'epsilon_l2': 60}},
                        {'attack_algorithm': 'bim', 'attack_config': {'alpha': 0.01, 'eps': 0.1, 'max_iter': 20, 'epsilon_l2': 60}},
                        {'attack_algorithm': 'pgd', 'attack_config': {'alpha': 0.01, 'eps': 0.1, 'max_iter': 20, 'epsilon_l2': 60}},
                        {'attack_algorithm': 'deepfool', 'attack_config': {'num_classes': 4, 'overshoot': 0.005, 'max_iter': 20, 'epsilon_l2': 60}},
                        {'attack_algorithm': 'cw2', 'attack_config': {'epsilon_l2': 60}},]

    # # between different interconnections
    # interconnection = 'b'
    # surrogate_interconnection = 'c'
    # for attack_info in attack_info_list[:]:
    #     attack_algorithm = attack_info['attack_algorithm']
    #     attack_config = attack_info['attack_config']
    #     for model_name in model_name_list[:]:
    #         attack_success_rate = perform_attack_transfer(interconnection, model_name,
    #                                                       surrogate_interconnection, model_name,
    #                                                       attack_algorithm, attack_config, device)
    #         print(interconnection, model_name, surrogate_interconnection, model_name, attack_algorithm, attack_config, device)
    #         print(attack_success_rate)

    # # between different interconnections
    # interconnection = 'c'
    # surrogate_interconnection = 'b'
    # for attack_info in attack_info_list[:]:
    #     attack_algorithm = attack_info['attack_algorithm']
    #     attack_config = attack_info['attack_config']
    #     for model_name in model_name_list[:]:
    #         attack_success_rate = perform_attack_transfer(interconnection, model_name,
    #                                                       surrogate_interconnection, model_name,
    #                                                       attack_algorithm, attack_config, device)
    #         print(interconnection, model_name, surrogate_interconnection, model_name, attack_algorithm, attack_config, device)
    #         print(attack_success_rate)

    for interconnection in interconnection_list[:1]:
        for attack_info in attack_info_list[:]:
            attack_algorithm = attack_info['attack_algorithm']
            attack_config = attack_info['attack_config']
            exp_name = f"ic_{interconnection}_{attack_algorithm}"
            # List of models
            model_list = []
            for model_name in model_name_list[:]:
                model_list.append((interconnection, model_name))
            transfer_experiments(exp_name, model_list, model_list, attack_algorithm, attack_config, device)
