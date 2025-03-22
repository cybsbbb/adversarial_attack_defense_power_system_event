import json
from pathlib import Path
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds

script_path = Path(__file__).resolve().parent


def get_exp_res(interconnection, model_name, attack_algorithm, max_queries, epsilon_l2):
    setup_random_seeds(428)
    # Setup save path
    script_dir = Path(__file__).resolve().parent
    result_dir = (f"{script_dir}/../../adv_exp_result/black/max_queries_{max_queries}_epsilon_l2_{epsilon_l2}/"
                  f"{interconnection}/{model_name}/{attack_algorithm}")
    exp_name = f"{interconnection}_{model_name}_{attack_algorithm}"
    result_sub_dir = f"{result_dir}/{exp_name}"
    with open(f'{result_sub_dir}/attack_res.json', 'r') as fp:
        information = json.load(fp)
    return information
