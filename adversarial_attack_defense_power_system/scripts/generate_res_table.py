import csv
from pathlib import Path
from adversarial_attack_defense_power_system.scripts.utils import get_exp_res

script_path = Path(__file__).resolve().parent


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent

    interconnection_list = ['b', 'c']
    # model_name_list = ['vgg13', 'mobilenet_v2', 'efficientnet', 'densenet121', 'resnet18', 'resnet50']
    model_name_list = ['vgg13', 'mobilenet_v2', 'densenet121', 'resnet50']
    attack_algorithms = ['simba_attack', 'zo_sign_sgd_attack', 'sign_hunter_attack',
                         'boundary_attack', 'opt_attack', 'sign_opt_attack',
                         'bit_schedule_v4', 'bit_schedule_v6',
                         'bit_schedule_transfer_v1', 'bit_schedule_transfer_v2']
    max_queries = 5000
    epsilon_l2 = 40

    for interconnection in interconnection_list[:1]:
        with open(f'{script_dir}/../../adv_exp_result/black/max_queries_{max_queries}_epsilon_l2_{epsilon_l2}/'
                  f'{interconnection}_result_table_success_rate.csv', 'w', newline='') as csvfile:
            fieldnames = ['model_name']
            for attack_algorithm in attack_algorithms:
                fieldnames.append(f'{attack_algorithm}')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for model_name in model_name_list[:]:
                row = {'model_name': model_name,}
                for attack_algorithm in attack_algorithms[:]:
                    information = get_exp_res(interconnection=interconnection,
                                              model_name=model_name,
                                              attack_algorithm=attack_algorithm,
                                              max_queries=max_queries,
                                              epsilon_l2=epsilon_l2)
                    row[attack_algorithm] = information['attack_success_rate']
                writer.writerow(row)

    for interconnection in interconnection_list[:1]:
        with open(f'{script_dir}/../../adv_exp_result/black/max_queries_{max_queries}_epsilon_l2_{epsilon_l2}/'
                  f'{interconnection}_result_table_average_query_cnt.csv', 'w', newline='') as csvfile:
            fieldnames = ['model_name']
            for attack_algorithm in attack_algorithms:
                fieldnames.append(f'{attack_algorithm}')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for model_name in model_name_list[:]:
                row = {'model_name': model_name,}
                for attack_algorithm in attack_algorithms[:]:
                    information = get_exp_res(interconnection=interconnection,
                                              model_name=model_name,
                                              attack_algorithm=attack_algorithm,
                                              max_queries=max_queries,
                                              epsilon_l2=epsilon_l2
                                              )
                    n = len(information['query_cnt'])
                    query_cnt_tot = 0
                    for i in range(n):
                        # if information['success'][i] is False:
                        #     query_cnt_tot += 1000
                        # else:
                        query_cnt_tot += min(information['query_cnt'][i], 1000)
                    query_cnt_mean = query_cnt_tot / n
                    row[attack_algorithm] = query_cnt_mean
                writer.writerow(row)

