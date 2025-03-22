import torch
import numpy as np
from adversarial_attack_defense_power_system.attacks.attacks_black.utils import get_probs, get_label
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds

def simba_attack(model, input_image, input_label, attack_config, targeted=False):
    input_image = input_image.cpu().numpy()
    input_label = input_label.cpu().numpy()
    # If the sample is original wrong prediction
    if get_label(model, input_image) != input_label.argmax(axis=-1)[0]:
        print("Original Wrong Prediction!")
        return torch.tensor(input_image), False, 1
    # Get the config variables
    max_queries = attack_config['max_queries'] if 'max_queries' in attack_config else 10000
    epsilon_l2 = attack_config['epsilon_l2'] if 'epsilon_l2' in attack_config else 20
    # TODO: make it better
    if max_queries == 1000:
        epsilon = 0.8
    elif max_queries == 5000:
        epsilon = 0.2
    elif max_queries == 10000:
        epsilon = 0.1
    else:
        epsilon = 0.05
    # Start the attack
    print(f"Starting Simba Attack, with max_queries: {max_queries}, epsilon_l2: {epsilon_l2}, epsilon: {epsilon}.")
    setup_random_seeds()
    x, y = np.copy(input_image), input_label.argmax(axis=-1)[0]
    # Number of the dim and query counter
    n_dims = 360 * 40 * 4
    query_cnt = 0
    perm = np.random.permutation(n_dims)
    last_prob = get_probs(model, x, y)
    for i in range(n_dims):
        diff = np.zeros(n_dims)
        diff[perm[i]] = epsilon
        left_prob = get_probs(model, x - diff.reshape(x.shape), y)
        query_cnt += 1
        if targeted != (left_prob < last_prob):
            x = x - diff.reshape(x.shape)
            last_prob = left_prob
        else:
            right_prob = get_probs(model, x + diff.reshape(x.shape), y)
            query_cnt += 1
            if targeted != (right_prob < last_prob):
                x = x + diff.reshape(x.shape)
                last_prob = right_prob
        perturbation = x - input_image
        perturbation_norm = np.linalg.norm(perturbation)
        # If successful attack
        if get_label(model, x) != y:
            perturbation = x - input_image
            print("Successfull Attack!")
            print(f"Prediction: {get_probs(model, x, y)}, Perturbation Norm: {np.linalg.norm(perturbation)}, Query Number: {query_cnt}.")
            return torch.tensor(x), True, query_cnt
        # Print interval results
        if i % 1000 == 0:
            print(f"Interation: {i}, Prediction: {get_probs(model, x, y)}, Perturbation Norm: {perturbation_norm}, Query Number: {query_cnt}.")
        # Break the attack if reach to query limitation or reach to epsilon_l2
        if query_cnt >= max_queries:
            print(f"Attack break because more than {max_queries} query!")
            break
        if perturbation_norm >= epsilon_l2:
            print("Too large perturbation, break the attack")
            break
    perturbation = x - input_image
    print("Failed Attack!")
    print(f"Prediction: {get_probs(model, x, y)}, Perturbation Norm: {np.linalg.norm(perturbation)}, Query Number: {query_cnt}.")
    return torch.tensor(x), False, query_cnt
