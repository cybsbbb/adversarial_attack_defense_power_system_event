import os
import csv
import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from adversarial_attack_defense_power_system.classifiers.load_classifier import load_classifier
from adversarial_attack_defense_power_system.scripts.generate_res_table import get_exp_res

script_path = Path(__file__).resolve().parent


def get_successful_rate_curve(interconnection, model_name, attack_algorithm, max_queries=5000, epsilon_l2=40):
    print(f"Get the successful rate curve, interconnection: {interconnection}, "
          f"model_name: {model_name}, attack_algorithm: {attack_algorithm}")
    # get exp result
    information = get_exp_res(interconnection=interconnection,
                              model_name=model_name,
                              attack_algorithm=attack_algorithm,
                              max_queries=max_queries,
                              epsilon_l2=epsilon_l2)
    success = information['success']
    query_cnt = information['query_cnt']
    n = len(success)

    max_queries = 1000
    query_intervals = list(range(0, max_queries + 1, 10))
    query_intervals_stat = [0] * len(query_intervals)

    for idx in range(n):
        if success[idx] is True and query_cnt[idx] <= max_queries:
            query_cnt = information['query_cnt']
            query_cnt[idx] = min(query_cnt[idx], max_queries)
            print(f"Successful sample, idx: {idx}, query_cnt: {query_cnt[idx]}")
            query_intervals_stat[(query_cnt[idx] - 1) // 10 + 1] += 1

    for i in range(1, len(query_intervals_stat)):
        query_intervals_stat[i] += query_intervals_stat[i - 1]

    print(query_intervals_stat)
    analysis_result_dir = f"{script_path}/../../adv_exp_result/analysis/successful_rate_curve"
    if not os.path.exists(analysis_result_dir):
        os.makedirs(analysis_result_dir)
    print(f"Saving the stat result to: {analysis_result_dir}/success_curve_{interconnection}_{model_name}_{attack_algorithm}.json.")
    with open(f'{analysis_result_dir}/success_curve_{interconnection}_{model_name}_{attack_algorithm}.json', 'w') as fp:
        json.dump(query_intervals_stat, fp)
    # plt.plot(query_intervals, query_intervals_stat)
    # plt.show()
    return 0


def plot_successful_curves(interconnection, model_name):
    # plt.figure(figsize=(15, 20))
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    def convert_ax2(ax1):
        y1, y2 = ax1.get_ylim()
        ax2.set_ylim(y1 * 100 / 483.0, y2 * 100 / 483.0)
        ax2.figure.canvas.draw()
    ax1.callbacks.connect("ylim_changed", convert_ax2)

    max_queries = 1000
    epsilon_l2 = 40
    attack_algorithms = ['simba_attack', 'zo_sign_sgd_attack', 'sign_hunter_attack',
                         'boundary_attack', 'opt_attack', 'sign_opt_attack',
                         # 'bit_schedule_v4', 'hybrid_attack_v2'
                         'bit_schedule_v6', 'bit_schedule_transfer_v2',]
    attack_algorithms_label = {'simba_attack': "SimBA",
                               'zo_sign_sgd_attack': "Zo-Sign-SGD",
                               'sign_hunter_attack': "SignHunter",
                               'boundary_attack': "BoundaryAttack",
                               'opt_attack': "OPT Attack",
                               'sign_opt_attack': "SignOPT Attack",
                               'bit_schedule_v6': "BitSchedule",
                               'bit_schedule_transfer_v2': "Ensemble-Based",
                               }

    query_intervals = list(range(0, max_queries + 1, 10))
    linestyles = ['-', 'dotted', 'dashed', 'dashdot']
    # for attack_algorithm in attack_algorithms[:]:
    for attack_algorithm_idx in range(len(attack_algorithms)):
        attack_algorithm = attack_algorithms[attack_algorithm_idx]
        analysis_result_dir = f"{script_path}/../../adv_exp_result/analysis/successful_rate_curve"
        with open(f'{analysis_result_dir}/success_curve_{interconnection}_{model_name}_{attack_algorithm}.json', 'r') as fp:
            query_intervals_stat = json.load(fp)
        # Set up the label
        label = attack_algorithms_label[attack_algorithm]
        linestyle = linestyles[attack_algorithm_idx % len(linestyles)]

        if attack_algorithm == "bit_schedule_transfer_v2":
            ax1.plot(query_intervals, query_intervals_stat, label=label, linestyle=linestyle, linewidth=3, color='black')
        else:
            ax1.plot(query_intervals, query_intervals_stat, label=label, linestyle=linestyle)
        # plt.plot(query_intervals, query_intervals_stat, label=label)

    analysis_figures_dir = f"{script_path}/../../adv_exp_result/analysis/analysis_figures"
    if not os.path.exists(analysis_figures_dir):
        os.makedirs(analysis_figures_dir)

    ax1.legend(fontsize=15, labelspacing=0.2, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.23))
    # ax1.set_title(f"Epsilon $L_2$: {epsilon_l2}", fontsize=15)
    ax1.set_xlabel("Query Number Limitation", fontsize=18)
    ax1.set_ylabel("Number of Successful Attack", fontsize=18)
    ax2.set_ylabel("Attack Success Rate (%)", fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    plt.savefig(f"{analysis_figures_dir}/{interconnection}_{model_name}_success_curve_{epsilon_l2}.png", dpi=150, bbox_inches='tight')
    # plt.show()
    # plt.close()
    return 0


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent

    interconnection_list = ['b', 'c']
    # model_name_list = ['vgg13', 'mobilenet_v2', 'efficientnet', 'densenet121', 'resnet18', 'resnet50']
    model_name_list = ['vgg13', 'mobilenet_v2', 'densenet121', 'resnet50']
    attack_algorithms = ['simba_attack', 'zo_sign_sgd_attack', 'sign_hunter_attack',
                         'boundary_attack', 'opt_attack', 'sign_opt_attack',
                         'bit_schedule_v4', 'bit_schedule_v6',
                         'bit_schedule_transfer_v1', 'bit_schedule_transfer_v2']

    # Calculate the successful rate for every exp
    for interconnection in interconnection_list[:1]:
        for attack_algorithm in attack_algorithms[:]:
            for model_name in model_name_list[:]:
                get_successful_rate_curve(interconnection, model_name, attack_algorithm)

    # Plot the successful rate curve figure
    for interconnection in interconnection_list[:1]:
        for model_name in model_name_list[:]:
            plot_successful_curves(interconnection, model_name)

