import math
import torch
import numpy as np
from adversarial_attack_defense_power_system.attacks.attacks_black.utils import get_probs, get_label
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds


def sign_hunter_attack(model, input_image, input_label, attack_config):
    input_image = input_image.cpu().numpy()
    input_label = input_label.cpu().numpy()
    if get_label(model, input_image) != input_label.argmax(axis=-1)[0]:
        print("Original Wrong Prediction!")
        return torch.tensor(input_image), False, 1
    # Get the config variables
    max_queries = attack_config['max_queries'] if 'max_queries' in attack_config else 10000
    epsilon_l2 = attack_config['epsilon_l2'] if 'epsilon_l2' in attack_config else 20
    # TODO: make it better
    # epsilon = 0.075 if epsilon_l2 == 20 else 0.04
    epsilon = 0.08 if epsilon_l2 == 40 else 0.04
    print(f"Starting SignHunter Attack, with max_queries: {max_queries}, epsilon_l2: {epsilon_l2}, epsilon: {epsilon}.")
    setup_random_seeds()
    # init the variables
    x0, y0 = np.copy(input_image), input_label.argmax(axis=-1)[0]
    x_shape = x0.shape
    dim = np.prod(x_shape[1:])
    query_cnt = 0
    # First step
    h, i = 0, 0
    sign_t = np.sign(np.ones(dim))
    # sign_t = np.sign(np.random.randn(dim))
    fxs_t = x0 + epsilon * sign_t.reshape(x_shape)
    cur_prob = get_probs(model, fxs_t, y0)
    best_prob = cur_prob
    query_cnt += 1
    # iteration flip
    for iter in range(max_queries):
        # chunk size
        chunk_len = (dim - 1) // (2 ** h) + 1
        # start and end index of the chunk
        istart = i * chunk_len
        iend = min(dim, (i + 1) * chunk_len)
        # flip the chunk
        sign_t[istart: iend] *= -1
        # check current probs changes
        fxs_t = x0 + epsilon * sign_t.reshape(x_shape)
        cur_prob = get_probs(model, fxs_t, y0)
        query_cnt += 1
        # Update the best_est_deriv
        if cur_prob >= best_prob:
            sign_t[istart: iend] *= -1
        else:
            best_prob = cur_prob
        # Current result!
        xt = x0 + epsilon * sign_t.reshape(x_shape)
        perturbation = epsilon * sign_t.reshape(x_shape)
        # If Successful attack
        if get_label(model, xt) != y0:
            print("Successful Attack!")
            print(f"Prediction: {get_probs(model, xt, y0)}, Perturbation Norm: {np.linalg.norm(perturbation)}, Query Number: {query_cnt}.")
            return torch.tensor(xt), True, query_cnt
        # Print interval results
        if iter % 1000 == 0:
            print(f"Iteration: {iter}, Predication: {get_probs(model, xt, y0)}, perturbation_norm: {np.linalg.norm(perturbation)}, Query_cnt: {query_cnt}.")
        # Break the attack if reach to query limitation.
        if query_cnt >= max_queries:
            print(f"Attack break because more than {max_queries} query!")
            break
        # Update the h and i
        i += 1
        if i == 2 ** h or iend == dim:
            h += 1
            i = 0
            if h == math.ceil(math.log2(dim)) + 1:
                print("New Process, reset x0!")
                x0 = np.copy(xt)
                h = 0
    xt = x0 + epsilon * sign_t.reshape(x_shape)
    perturbation = xt - input_image
    print("Failed Attack!")
    print(f"Prediction: {get_probs(model, xt, y0)}, Perturbation Norm: {np.linalg.norm(perturbation)}, Query Number: {query_cnt}.")
    return torch.tensor(xt), False, query_cnt
