import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from adversarial_attack_defense_power_system.classifiers.load_classifier import load_classifier
from adversarial_attack_defense_power_system.dataset_loader.pmu_event_dataset import PMUEventDataset
from adversarial_attack_defense_power_system.attacks.attack_wrapper_black import attack_on_dataloader_black
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds

script_path = Path(__file__).resolve().parent


def perform_attack_black_box(interconnection, model_name, attack_algorithm, attack_config, device):
    setup_random_seeds(428)
    # Setup save path
    script_dir = Path(__file__).resolve().parent
    max_queries = attack_config['max_queries']
    epsilon_l2 = attack_config['epsilon_l2']
    result_dir = (f"{script_dir}/../../adv_exp_result/black/max_queries_{max_queries}_epsilon_l2_{epsilon_l2}/"
                  f"{interconnection}/{model_name}/{attack_algorithm}")

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

    # Load the trained surrogate classifier
    surrogate_net = load_classifier(interconnection=interconnection, model_name=attack_config['surrogate_model_name'], device=device)
    surrogate_net.eval()
    attack_config['surrogate_net'] = surrogate_net

    # Perform attack
    start_time = time.time()
    x, x_adv, success, query_cnt = attack_on_dataloader_black(testloader, net, attack_algorithm, attack_config, device)
    end_time = time.time()

    # Stat the attack success rate and average query
    n = len(success)
    attack_success_rate = sum(map(int, success)) / n
    query_cnt_mean = sum(map(int, query_cnt)) / n
    perturbation = x_adv - x
    perturbation_norm = torch.mean(torch.linalg.vector_norm(torch.flatten(perturbation, start_dim=1), dim=1)).detach().cpu().item()
    print(f"perturbation_norm: {perturbation_norm}")

    # Save result
    elapsed_time = end_time - start_time
    attack_config['surrogate_net'] = None
    information = {"attack_algorithm": attack_algorithm,
                   "attack_config": attack_config,
                   "elapsed_time": elapsed_time,
                   "attack_success_rate": attack_success_rate,
                   "perturbation_norm": perturbation_norm,
                   "query_cnt_mean": query_cnt_mean,
                   "success": success,
                   "query_cnt": query_cnt,
                   }
    print(information)
    np.save(f'{result_sub_dir}/x_adv.npy', x_adv.detach().cpu().numpy())
    np.save(f'{result_sub_dir}/x.npy', x.detach().cpu().numpy())
    with open(f'{result_sub_dir}/attack_res.json', 'w') as fp:
        json.dump(information, fp, indent=4)
    print(f"exp_name: {exp_name}, attack_success_rate: {attack_success_rate}")
    return 0


if __name__ == '__main__':
    # Set up the device
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")
    # device = torch.device("cpu")

    interconnection_list = ['b', 'c']
    # model_name_list = ['vgg13', 'mobilenet_v2', 'efficientnet', 'densenet121', 'resnet18', 'resnet50']
    model_name_list = ['vgg13', 'mobilenet_v2', 'densenet121', 'resnet50']
    attack_algorithms = ['simba_attack', 'zo_sign_sgd_attack', 'sign_hunter_attack',
                         'boundary_attack', 'opt_attack', 'sign_opt_attack',
                         'bit_schedule_v4', 'bit_schedule_v6',
                         'bit_schedule_transfer_v1', 'bit_schedule_transfer_v2']
    attack_config = {'max_queries': 5000, 'epsilon_l2': 20, 'surrogate_model_name': 'resnet18'}

    for interconnection in interconnection_list[:1]:
        attack_config['epsilon_l2'] = 40 if interconnection == 'b' else 80
        for model_name in model_name_list[:]:
            for attack_algorithm in attack_algorithms[:]:
                perform_attack_black_box(interconnection=interconnection,
                                         model_name=model_name,
                                         attack_algorithm=attack_algorithm,
                                         attack_config=attack_config,
                                         device=device)
                torch.cuda.empty_cache()
