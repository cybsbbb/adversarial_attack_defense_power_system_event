import os
import csv
import torch
import numpy as np
from pathlib import Path
from adversarial_attack_defense_power_system.classifiers.load_classifier import load_classifier
from adversarial_attack_defense_power_system.classifiers.evaluation import evaluation_numpy
from adversarial_attack_defense_power_system.dataset_loader.pmu_event_dataset import PMUEventDataset
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds
from adversarial_attack_defense_power_system.defenses.input_transformation.input_transformation_wrapper import input_transformation_wrapper


def defense_purification(x_adv, input_transformation_algorithm, transformation_parameters):
    # Input transformation
    x_adv = np.transpose(x_adv, (0, 2, 3, 1))
    x_adv_purified = input_transformation_wrapper(x_adv, input_transformation_algorithm, transformation_parameters)
    x_adv_purified = np.transpose(x_adv_purified, (0, 3, 1, 2))
    return x_adv_purified


def diffusion_purification_benchmark_black(interconnection, model_name, attack_algorithm,
                                           input_transformation_algorithm, transformation_parameters, device):
    setup_random_seeds(428)
    script_dir = Path(__file__).resolve().parent
    # Get the attack result dir, load attacked data
    result_dir = (f"{script_dir}/../../adv_exp_result/black/max_queries_5000_epsilon_l2_40/"
                  f"{interconnection}/{model_name}/{attack_algorithm}")
    exp_name = f"{interconnection}_{model_name}_{attack_algorithm}"
    result_sub_dir = f"{result_dir}/{exp_name}"
    x_adv = np.load(f'{result_sub_dir}/x_adv.npy')
    # Load the dataset
    testset = PMUEventDataset(interconnection=interconnection, train=False)
    # Load the trained classifier
    net = load_classifier(interconnection=interconnection, model_name=model_name, device=device)
    # Diffusion adversarial purification
    x_adv_purified = defense_purification(x_adv, input_transformation_algorithm, transformation_parameters)
    # Save purified result
    np.save(f'{result_sub_dir}/x_adv_purified.npy', x_adv_purified)
    # Evaluate the F1 score
    f1_after = evaluation_numpy(x_adv_purified, testset.label, net, device)
    print(f"F1 after: {f1_after}")
    return f1_after


if __name__ == '__main__':
    # Set up the device
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")

    script_dir = Path(__file__).resolve().parent

    interconnection_list = ['b', 'c']
    model_name_list = ['vgg13', 'mobilenet_v2', 'densenet121', 'resnet18', 'resnet50']
    attack_algorithms = ['simba_attack', 'zo_sign_sgd_attack', 'sign_hunter_attack',
                         'boundary_attack', 'opt_attack', 'sign_opt_attack',
                         'bit_schedule_v4', 'bit_schedule_v6',
                         'bit_schedule_transfer_v1', 'bit_schedule_transfer_v2']
    input_transformation_algorithms = ['', 'spatial_smoothing', 'low_pass_filtering', 'feature_squeezing',
                                       'svd_decomposition', 'event_decomposition', 'diffusion']


    for interconnection in interconnection_list[:1]:
        result_file_dir = f"./../../adv_exp_result/analysis/defense_benchmark"
        if not os.path.exists(result_file_dir):
            os.makedirs(result_file_dir)
        result_file_path = f'{result_file_dir}/{interconnection}_result_table_defense_benchmark.csv'
        with open(result_file_path, 'w', newline='') as csvfile:
            fieldnames = ['exp_name']
            for attack_algorithm in attack_algorithms:
                fieldnames.append(f'{attack_algorithm}')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for model_name in model_name_list[:]:
                for input_transformation_algorithm in input_transformation_algorithms[:]:
                    exp_name = f'ic_{interconnection}_{model_name}_{input_transformation_algorithm}'
                    row = {'exp_name': exp_name}
                    transformation_parameters = ""
                    if input_transformation_algorithm == 'diffusion':
                        transformation_parameters = {'interconnection': interconnection,
                                                     'timesteps': 20,
                                                     'beta_type': 'linear',
                                                     'transformation_type': 'ddim',
                                                     'diffusion_t': 0.1,
                                                     'denoise_steps': 3}
                    for attack_algorithm in attack_algorithms[:]:
                        f1_after = diffusion_purification_benchmark_black(interconnection=interconnection,
                                                                          model_name=model_name,
                                                                          attack_algorithm=attack_algorithm,
                                                                          input_transformation_algorithm=input_transformation_algorithm,
                                                                          transformation_parameters=transformation_parameters,
                                                                          device=device)
                        row[attack_algorithm] = f1_after
                        torch.cuda.empty_cache()
                    writer.writerow(row)
