import torch
import numpy as np
from adversarial_attack_defense_power_system.attacks.attacks_black.utils import get_probs, get_label
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds


def zo_sign_sgd_attack(model, input_image, input_label, attack_config, q=25, fd_eta=0.05):
    input_image = input_image.cpu().numpy()
    input_label = input_label.cpu().numpy()
    # Check if the original wrong prediction!
    if get_label(model, input_image) != input_label.argmax(axis=-1)[0]:
        print("Original Wrong Prediction!")
        return torch.tensor(input_image), False, 1
    # Get the config variables
    max_queries = attack_config['max_queries'] if 'max_queries' in attack_config else 10000
    epsilon_l2 = attack_config['epsilon_l2'] if 'epsilon_l2' in attack_config else 20
    # Setup the learning rate
    # lr = 0.01 if epsilon_l2 == 20 else 0.005
    lr = 0.01
    print(f"Starting zo_sign_sgd Attack, with max_queries: {max_queries}, epsilon_l2: {epsilon_l2}, learn_rate: {lr}")
    setup_random_seeds()
    x, y = np.copy(input_image), input_label.argmax(axis=-1)[0]
    shape = x.shape
    query_cnt = 0
    num_axes = len(shape[1:])
    gs_t = np.zeros_like(x)
    T = max_queries
    for iter in range(T):
        for _ in range(q):
            exp_noise = np.random.randn(*x.shape)
            fxs_t = x + fd_eta * exp_noise
            bxs_t = x
            est_deriv = (get_probs(model, fxs_t, y) - get_probs(model, bxs_t, y)) / fd_eta
            query_cnt += 2
            gs_t += est_deriv.reshape(-1, *[1] * num_axes) * exp_noise
        # Sign Step
        x = x - lr * np.sign(gs_t)
        # L2 step (Use Sign step now)
        # x = x - lr * gs_t / q
        # Make sure the perturbation within the epsilon_l2
        perturbation = x - input_image
        perturbation_norm = np.linalg.norm(perturbation)
        if perturbation_norm > epsilon_l2:
            perturbation = epsilon_l2 * perturbation / (perturbation_norm + 0.001)
            x = input_image + perturbation
        # If successful attack
        if get_label(model, x) != y:
            print("Successfull Attack!")
            print(f"Prediction: {get_probs(model, x, y)}, Perturbation Norm: {np.linalg.norm(perturbation)}, Query Number: {query_cnt}.")
            return torch.tensor(x), True, query_cnt
        # Print interval results
        if iter % 20 == 0:
            print(
                f"Iteration: {iter}, Queries: {query_cnt}, Prediction: {get_probs(model, x, y)}, Perturbation norm: {np.linalg.norm(perturbation)}.")
        # Break the attack if reach to query limitation.
        if query_cnt >= max_queries:
            print(f"Attack break because more than {max_queries} query!")
            break
    perturbation = x - input_image
    print("Failed Attack!")
    print(f"Prediction: {get_probs(model, x, y)}, Perturbation Norm: {np.linalg.norm(perturbation)}, Query Number: {query_cnt}.")
    return torch.tensor(x), False, query_cnt
