import os
import time
import json
import numpy as np
import torch
from adversarial_attack_defense_power_system.classifiers.load_classifier import load_classifier
from adversarial_attack_defense_power_system.dataset_loader.pmu_event_dataset import PMUEventDataset
from adversarial_attack_defense_power_system.attacks.attack_wrapper_white import attack_on_dataloader_white
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds
from pathlib import Path


def perform_attack_white(interconnection, model_name, attack_algorithm, attack_config, device):
    setup_random_seeds(428)
    # Setup save path
    script_dir = Path(__file__).resolve().parent
    result_dir = f"{script_dir}/../../adv_exp_result/white/{interconnection}/{model_name}/{attack_algorithm}"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    exp_name = f"{interconnection}_{model_name}_{attack_algorithm}"
    result_sub_dir = f"{result_dir}/{exp_name}"
    print(result_sub_dir)
    if not os.path.exists(result_sub_dir):
        os.makedirs(result_sub_dir)

    # Load the dataset
    testset = PMUEventDataset(interconnection=interconnection, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Load the trained classifier
    net = load_classifier(interconnection=interconnection, model_name=model_name, device=device)

    # Perform attack
    start_time = time.time()
    x, x_adv, success = attack_on_dataloader_white(testloader, net, attack_algorithm, attack_config, device)
    end_time = time.time()

    # Stat the attack success rate and average query
    n = len(success)
    attack_success_rate = sum(map(int, success)) / n
    perturbation = x_adv - x
    perturbation_norm = torch.mean(torch.linalg.vector_norm(torch.flatten(perturbation, start_dim=1), dim=1)).detach().cpu().item()
    print(f"perturbation_norm: {perturbation_norm}")

    # Save result
    elapsed_time = end_time - start_time
    information = {"attack_algorithm": attack_algorithm,
                   "attack_config": attack_config,
                   "elapsed_time": elapsed_time,
                   "attack_success_rate": attack_success_rate,
                   "perturbation_norm": perturbation_norm,
                   "success": success,
                   }
    np.save(f'{result_sub_dir}/x_adv.npy', x_adv.detach().cpu().numpy())
    np.save(f'{result_sub_dir}/x.npy', x.detach().cpu().numpy())
    with open(f'{result_sub_dir}/attack_res.json', 'w') as fp:
        json.dump(information, fp, indent=4)
    print(f"exp_name: {exp_name}, attack_success_rate: {attack_success_rate}")
    return 0


if __name__ == '__main__':
    # Set up the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")

    interconnection_list = ['b', 'c']
    model_name_list = ['vgg13', 'mobilenet_v2', 'efficientnet', 'densenet121', 'resnet18', 'resnet50']
    attack_info_list = [{'attack_algorithm': 'fgsm', 'attack_config': {'epsilon': 0.05}},
                        {'attack_algorithm': 'bim', 'attack_config': {'alpha': 0.005, 'eps': 0.05, 'max_iter': 20}},
                        {'attack_algorithm': 'pgd', 'attack_config': {'alpha': 0.005, 'eps': 0.05, 'max_iter': 20}},
                        {'attack_algorithm': 'deepfool', 'attack_config': {'num_classes': 4, 'overshoot': 0.005, 'max_iter': 20}},
                        {'attack_algorithm': 'cw2', 'attack_config': None},
                        ]

    for interconnection in interconnection_list[:]:
        for model_name in model_name_list[:]:
            for attack_info in attack_info_list[:]:
                attack_algorithm = attack_info['attack_algorithm']
                attack_config = attack_info['attack_config']
                perform_attack_white(interconnection=interconnection,
                                     model_name=model_name,
                                     attack_algorithm=attack_algorithm,
                                     attack_config=attack_config,
                                     device=device)
                torch.cuda.empty_cache()
