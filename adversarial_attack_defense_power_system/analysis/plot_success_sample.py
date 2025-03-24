import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from adversarial_attack_defense_power_system.classifiers.load_classifier import load_classifier
import matplotlib.pyplot as plt


def visualization(event, save_path='', isSave=False):
    plt.style.context(['ieee', 'no-latex'])

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.55)
    plt.tick_params(labelsize=20)

    axes[0].set_title("Real Power (Z-Score)", fontsize=30)
    axes[0].plot(event[:, :, 0])
    axes[0].tick_params(labelsize=25)
    axes[0].set_xlabel('Timestamp', fontsize=20)
    axes[0].xaxis.set_label_coords(0.93, -0.25)

    axes[1].set_title("Reactive Power (Z-Score)", fontsize=30)
    axes[1].plot(event[:, :, 1])
    axes[1].tick_params(labelsize=25)
    axes[1].set_xlabel('Timestamp', fontsize=20)
    axes[1].xaxis.set_label_coords(0.93, -0.25)

    axes[2].set_title("Voltage Magnitude (Z-Score)", fontsize=30)
    axes[2].plot(event[:, :, 2])
    axes[2].tick_params(labelsize=25)
    axes[2].set_xlabel('Timestamp', fontsize=20)
    axes[2].xaxis.set_label_coords(0.93, -0.25)

    axes[3].set_title("Frequency (Z-Score)", fontsize=30)
    axes[3].plot(event[:, :, 3])
    axes[3].tick_params(labelsize=25)
    axes[3].set_xlabel('Timestamp', fontsize=20)
    axes[3].xaxis.set_label_coords(0.93, -0.25)

    axes[0].set_ylim([-6, 6])
    axes[1].set_ylim([-6, 6])
    axes[2].set_ylim([-6, 6])
    axes[3].set_ylim([-4, 4])

    if isSave == True:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
    return 0


if __name__ == '__main__':
    script_path = Path(__file__).resolve().parent

    # Load the data
    x = np.load('../../data/datasets/ic_b/test_data.npy')[:, :, :40, :]
    x_adv = np.load('../../adv_exp_result/black/max_queries_1000_epsilon_l2_40'
                    '/b/resnet50/bit_schedule_v6/b_resnet50_bit_schedule_v6/x_adv.npy')
    x_adv = np.transpose(x_adv, (0, 2, 3, 1))

    # Get the samples
    # target_idx = 143
    target_idx = 109
    original_sample = x[target_idx]
    adx_sample = x_adv[target_idx]
    perturbation = adx_sample - original_sample

    # Load the trained classifier
    net = load_classifier(interconnection='b', model_name='resnet50', device='cpu')
    net.eval()

    original_sample_torch = torch.tensor(np.transpose(original_sample[None,], (0, 3, 1, 2)), dtype=torch.float)
    adv_sample_torch = torch.tensor(np.transpose(adx_sample[None,], (0, 3, 1, 2)), dtype=torch.float)

    softmax = nn.Softmax(dim=1)
    print(f"Original prediction: {softmax(net(original_sample_torch))}")
    print(f"Adversarial sample prediction: {softmax(net(adv_sample_torch))}")

    sample_figures_dir = f"./../../adv_exp_result/analysis/sample_figures"
    if not os.path.exists(sample_figures_dir):
        os.makedirs(sample_figures_dir)

    visualization(original_sample, f"{sample_figures_dir}/original_sample.png", isSave=True)
    visualization(adx_sample, f"{sample_figures_dir}/adx_sample.png", isSave=True)
    visualization(perturbation, f"{sample_figures_dir}/perturbation.png", isSave=True)
