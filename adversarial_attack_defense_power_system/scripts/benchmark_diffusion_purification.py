import csv
import torch
import numpy as np
from pathlib import Path
from adversarial_attack_defense_power_system.classifiers.load_classifier import load_classifier
from adversarial_attack_defense_power_system.classifiers.evaluation import evaluation_numpy
from adversarial_attack_defense_power_system.dataset_loader.pmu_event_dataset import PMUEventDataset
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds
from adversarial_attack_defense_power_system.defenses.diffusion.diffusion_tranformation_wrapper import diffusion_dataset_transformation


def diffusion_purification(x_adv):
    transformation_parameters = {'interconnection': 'b',
                                 'timesteps': 20,
                                 'beta_type': 'linear',
                                 'transformation_type': 'ddim',
                                 'diffusion_t': 0.1,
                                 'denoise_steps': 3}
    x_adv = np.transpose(x_adv, (0, 2, 3, 1))
    x_adv_purified = diffusion_dataset_transformation(x_adv, transformation_parameters)
    x_adv_purified = np.transpose(x_adv_purified, (0, 3, 1, 2))
    return x_adv_purified


def diffusion_purification_benchmark_black(interconnection, model_name, attack_algorithm, device):
    setup_random_seeds(428)
    script_dir = Path(__file__).resolve().parent
    # Get the attack result dir, load attacked data
    result_dir = f"{script_dir}/../../adv_attack_result/black/{interconnection}/{model_name}/{attack_algorithm}"
    exp_name = f"{interconnection}_{model_name}_{attack_algorithm}"
    result_sub_dir = f"{result_dir}/{exp_name}"
    x_adv = np.load(f'{result_sub_dir}/x_adv.npy')
    # Load the dataset
    testset = PMUEventDataset(interconnection=interconnection, train=False)
    # Load the trained classifier
    net = load_classifier(interconnection=interconnection, model_name=model_name, device=device)
    # Diffusion adversarial purification
    x_adv_purified = diffusion_purification(x_adv)
    # Save purified result
    np.save(f'{result_sub_dir}/x_adv_purified.npy', x_adv_purified)

    f1_before = evaluation_numpy(x_adv, testset.label, net, device)
    print(f"F1 before: {f1_before:}")

    f1_after = evaluation_numpy(x_adv_purified, testset.label, net, device)
    print(f"F1 after: {f1_after}")

    return f1_before, f1_after


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
    model_name_list = ['vgg13', 'mobilenet_v2', 'densenet121', 'resnet18', 'resnet50']
    attack_algorithms = ['simba_attack', 'zo_sign_sgd_attack', 'sign_hunter_attack',
                         'boundary_attack', 'opt_attack', 'sign_opt_attack',
                         'bit_schedule_v6', 'bit_schedule_transfer_v2'][-2:]

    for interconnection in interconnection_list[:1]:
        script_dir = Path(__file__).resolve().parent
        result_file_path = f'{script_dir}/../../adv_attack_result/black/{interconnection}_result_table_diffusion_purification.csv'
        with open(result_file_path, 'w', newline='') as csvfile:
            fieldnames = ['model_name']
            for attack_algorithm in attack_algorithms:
                fieldnames.append(f'{attack_algorithm}')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for model_name in model_name_list[:]:
                row_before = {'model_name': model_name + "_before", }
                row_after = {'model_name': model_name + "_after", }
                for attack_algorithm in attack_algorithms[:]:
                    f1_before, f1_after = diffusion_purification_benchmark_black(interconnection=interconnection,
                                                                                 model_name=model_name,
                                                                                 attack_algorithm=attack_algorithm,
                                                                                 device=device)
                    row_before[attack_algorithm] = f1_before
                    row_after[attack_algorithm] = f1_after
                    torch.cuda.empty_cache()
                writer.writerow(row_before)
                writer.writerow(row_after)
