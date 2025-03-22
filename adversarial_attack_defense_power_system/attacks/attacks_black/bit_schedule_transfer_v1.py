import math
import torch
import numpy as np
from adversarial_attack_defense_power_system.attacks.attacks_black.utils import get_probs, get_label
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds
from adversarial_attack_defense_power_system.attacks.attacks_white.fgsm import get_data_grad
from adversarial_attack_defense_power_system.attacks.attacks_white.deepfool import deepfool_attack


def bit_schedule_attack_transfer_v1(model, input_image, input_label, attack_config):
    """
        This version only perform one query for each schedule
    """
    input_image = input_image.cpu().numpy()
    input_label = input_label.cpu().numpy()
    if get_label(model, input_image) != input_label.argmax(axis=-1)[0]:
        print("Original Wrong Prediction!")
        return torch.tensor(input_image), False, 1
    # Get the config variables
    max_queries = attack_config['max_queries'] if 'max_queries' in attack_config else 1000
    epsilon_l2 = attack_config['epsilon_l2'] if 'epsilon_l2' in attack_config else 20
    fd_eta = attack_config['fd_eta'] if 'fd_eta' in attack_config else 0.01
    T = attack_config['T'] if 'T' in attack_config else 1
    lr = 0.05
    surrogate_net = attack_config['surrogate_net']
    print(f"Starting Bit Schedule Attack transfer V1, with max_queries: {max_queries}, epsilon_l2: {epsilon_l2}, learning_rate: {lr}.")
    setup_random_seeds()
    # init the variables
    x, y = np.copy(input_image), input_label.argmax(axis=-1)[0]
    x_shape = x.shape
    dim = np.prod(x_shape[1:])
    bit_length = int(math.log2(dim)) + 1
    query_cnt = 0

    # First shoot by surrogate_net
    sample_x_adv, _ = deepfool_attack(torch.tensor(input_image, device=next(surrogate_net.parameters()).device),
                                      torch.tensor(input_label, device=next(surrogate_net.parameters()).device),
                                      surrogate_net)
    sample_x_adv = sample_x_adv.cpu().numpy()
    # Adjust the perturbation norm
    perturbation = sample_x_adv - input_image
    perturbation_norm = np.linalg.norm(perturbation)
    if perturbation_norm > 1e-5:
        perturbation = perturbation * (epsilon_l2 / perturbation_norm)
        sample_x_adv = input_image + perturbation
        if get_label(model, sample_x_adv) != y:
            print("Successfull Attack!")
            print(f"Prediction: {get_probs(model, sample_x_adv, y)}, "
                  f"Perturbation Norm: {np.linalg.norm(perturbation)}, "
                  f"Query Number: {query_cnt}.")
            query_cnt += 1
            return torch.tensor(sample_x_adv), True, query_cnt

    # Initiate the bit scheduler
    bits_idx_list = [[] for _ in range(bit_length)]
    for i in range(dim):
        for bit in range(bit_length):
            if (i >> bit) & 1 == 1:
                bits_idx_list[bit].append(i)
    bits_idx_list = [np.array(bits_idx_list[bit]) for bit in range(bit_length)]
    # Start the attack iterations!
    # best_perturbation = np.zeros_like(x)
    last_prob = get_probs(model, x, y)
    query_cnt += 1
    for iter in range(max_queries):
        # exp version
        degree = 2
        focus = math.exp(-degree * iter)
        if iter > 8:
            focus = 0
        # print(focus)
        # linear version
        # focus_low, focus_high = 0.0, 0.5
        # focus = (focus_high - focus_low) * max(0, (1 - 20 * query_cnt / max_queries)) + focus_low
        top_portion = 0.50 - (0.50 - 0.20) * (query_cnt / max_queries)
        gradient_est = np.zeros_like(x)
        for inner_iter in range(T):
            perm = np.random.permutation(dim)
            for bit in range(bit_length):
                bits_idx = bits_idx_list[bit]
                cur_sign = np.ones(dim)
                cur_sign[perm[bits_idx]] *= -1
                exp_noise = cur_sign.reshape(x_shape)
                # first direction
                x_t = x + fd_eta * exp_noise
                first_prob = get_probs(model, x_t, y)
                # second direction
                x_t_flip = x - fd_eta * exp_noise
                second_prob = get_probs(model, x_t_flip, y)
                # The weight is (first_prob - second_prob)
                gradient_est -= exp_noise * (first_prob - second_prob)
                query_cnt += 2
        # Only keep the top (top_portion) of the value in the gradient estimation
        gradient_flattened = gradient_est.flatten()
        num_to_zero = int(len(gradient_flattened) * (1 - top_portion))
        indices_to_zero = np.argpartition(np.abs(gradient_flattened), num_to_zero)[:num_to_zero]
        gradient_flattened[indices_to_zero] = 0
        gradient_est = gradient_flattened.reshape(gradient_est.shape)
        gradient_est_surrogate = get_data_grad(torch.tensor(x, device=next(surrogate_net.parameters()).device),
                                               torch.tensor(input_label, device=next(surrogate_net.parameters()).device),
                                               surrogate_net).cpu().numpy()

        # print(np.linalg.norm(gradient_est), np.linalg.norm(gradient_est_surrogate))
        # gradient_est = gradient_est
        gradient_est = (gradient_est / np.linalg.norm(gradient_est) +
                        focus * gradient_est_surrogate / np.linalg.norm(gradient_est_surrogate))

        # Update sample
        x = x + lr * np.sign(gradient_est)
        perturbation = x - input_image
        perturbation_norm = np.linalg.norm(perturbation)
        if perturbation_norm > epsilon_l2:
            perturbation = epsilon_l2 * perturbation / (perturbation_norm + 0.001)
        # Only Update the sample if this step decrease the prob
        cur_prob = get_probs(model, input_image + perturbation, y)
        query_cnt += 1
        # If the difference is too small, expand the search space.
        if abs(cur_prob - last_prob) < 1e-5 and fd_eta < epsilon_l2 / 160:
            fd_eta *= 2
            print(fd_eta)
        # if cur_prob < best_prob:
        #     best_perturbation = perturbation
        #     best_prob = cur_prob
        # else:
        #     perturbation = best_perturbation
        x = input_image + perturbation
        last_prob = cur_prob
        # Print the internal results
        if iter % 50 == 0:
            print(f"Current Iteration: {iter}, query number: {query_cnt}, top portion: {top_portion}.")
            print(f"Current prediction result: {get_probs(model, x, y)}, perturbation norm: {np.linalg.norm(perturbation)}")
        # If successfully attack
        if get_label(model, x) != y:
            print("Successfull Attack!")
            print(f"Prediction: {get_probs(model, x, y)}, Perturbation Norm: {np.linalg.norm(perturbation)}, Query Number: {query_cnt}.")
            return torch.tensor(x), True, query_cnt
        # Break the attack if reach to query limitation.
        if query_cnt >= max_queries:
            print(f"Attack break because more than {max_queries} query!")
            break
    perturbation = x - input_image
    print("Failed Attack!")
    print(f"Prediction: {get_probs(model, x, y)}, Perturbation Norm: {np.linalg.norm(perturbation)}, Query Number: {query_cnt}.")
    return torch.tensor(x), False, query_cnt
