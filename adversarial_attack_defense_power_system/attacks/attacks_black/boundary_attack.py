import torch
import numpy as np
from adversarial_attack_defense_power_system.attacks.attacks_black.utils import get_probs, get_label
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds


def is_adversarial(model, xt, y0):
    """
        Check if xt is a valid adversarial sample.
    """
    return get_label(model, xt) != y0


def _orthogonal_perturb(delta: float, current_sample: np.ndarray, original_sample: np.ndarray) -> np.ndarray:
    """
        Create an orthogonal perturbation.
    """
    # Generate perturbation randomly
    perturb = np.random.randn(*current_sample.shape)
    # Rescale the perturbation
    perturb /= np.linalg.norm(perturb)
    perturb *= delta * np.linalg.norm(original_sample - current_sample)
    # Project the perturbation onto sphere
    direction = original_sample - current_sample

    direction_flat = direction.flatten()
    perturb_flat = perturb.flatten()

    direction_flat /= np.linalg.norm(direction_flat)
    perturb_flat -= np.dot(perturb_flat, direction_flat.T) * direction_flat
    perturb = perturb_flat.reshape(current_sample.shape)

    hypotenuse = np.sqrt(1 + delta ** 2)
    perturb = ((1 - hypotenuse) * (current_sample - original_sample) + perturb) / hypotenuse
    return perturb


def _best_adv(original_sample: np.ndarray, potential_advs: np.ndarray) -> np.ndarray:
    """
        Return the best adversarial from the potential samples
    """
    shape = potential_advs.shape
    min_idx = np.linalg.norm(original_sample.flatten() - potential_advs.reshape(shape[0], -1), axis=1).argmin()
    return potential_advs[min_idx][None, ...]


def initialize(model, x0, y0, low_bound=-3.0, high_bound=3.0, max_search=1000):
    """
        Find the initialization of the boundary by the BlendedUniformNoiseAttack
        For the PMU data, the range of the data is set to [-3.0, 3.0]
    """
    success = False
    query_cnt = 0
    random_noise = None
    # Search a adversarial point from uniform distribution
    for i in range(max_search):
        random_noise = np.random.uniform(low_bound, high_bound, size=x0.shape)
        success = is_adversarial(model, random_noise, y0)
        query_cnt += 1
        if success:
            break
    # Cannot find an valid adversarial attack, return None and query_cnt
    if not success:
        print("Cannot find a adversarial example")
        return None, query_cnt
    # Binary search to minimize the l2 distance to original image.
    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0
        blended = (1 - mid) * x0 + mid * random_noise
        success = is_adversarial(model, blended, y0)
        query_cnt += 1
        if success:
            high = mid
        else:
            low = mid
    initialization = (1 - high) * x0 + high * random_noise

    return initialization, query_cnt


def boundary_attack(model, input_image, input_label, attack_config):
    """
        Boundary Attack
    """
    input_image = input_image.cpu().numpy()
    input_label = input_label.cpu().numpy()
    x0, y0 = input_image, input_label.argmax(axis=-1)[0]
    if get_label(model, x0) != y0:
        print("Original Wrong Prediction!")
        return torch.tensor(x0), False, 1
    # Get the config variables
    max_queries = attack_config['max_queries'] if 'max_queries' in attack_config else 10000
    epsilon_l2 = attack_config['epsilon_l2'] if 'epsilon_l2' in attack_config else 20
    print(f"Starting Boundary Attack, with max_queries: {max_queries}, epsilon_l2: {epsilon_l2}.")
    setup_random_seeds()
    # Initialize the x_adv
    x_adv, query_cnt = initialize(model, x0, y0)
    # perturbation = x_adv - x0
    if x_adv is None:
        print(f"Failed! Couldn't find valid initialize adversarial, Total Query Count: {query_cnt}.")
        return torch.tensor(x0), False, query_cnt

    # Setup the parameters (Ugly)
    curr_delta: float = 0.01
    curr_epsilon: float = 0.01
    num_trial: int = 25
    sample_size: int = 20
    step_adapt: float = 0.667
    min_epsilon: float = 0.0

    # Start the main loop of the Boundary Attack
    for iter in range(max_queries):
        # Trust region method to adjust delta
        for _ in range(num_trial):
            potential_advs_list = []
            for _ in range(sample_size):
                potential_adv = x_adv + _orthogonal_perturb(curr_delta, x_adv, x0)
                # Clip (no need for PMU data)
                # potential_adv = np.clip(potential_adv, -3.0, 3.0)
                potential_advs_list.append(potential_adv)
            # Get the ratio of the successful move
            # preds = model(np.concatenate(potential_advs_list, axis=0)).numpy().argmax(axis=-1)
            # preds = model(torch.Tensor(np.concatenate(potential_advs_list, axis=0))).detach().cpu().numpy().argmax(axis=-1)
            preds = model(torch.tensor(np.concatenate(potential_advs_list, axis=0),
                                       dtype=torch.float32,
                                       device=next(model.parameters()).device)).detach().cpu().numpy().argmax(axis=-1)
            query_cnt += len(potential_advs_list)
            satisfied = (preds != y0)
            delta_ratio = np.mean(satisfied)
            # Adjust the delta based on the ratio
            if delta_ratio < 0.2:
                curr_delta *= step_adapt
            elif delta_ratio > 0.5:
                curr_delta /= step_adapt
            # If found the valid adversarial candidate
            if delta_ratio > 0:
                x_advs = np.concatenate(potential_advs_list)[np.where(satisfied)[0]]
                break
        else:
            if query_cnt > max_queries:
                print(f"Attack break because more than {max_queries} query!")
                break
            continue

        # Trust region method to adjust epsilon
        for _ in range(num_trial):
            perturb = np.repeat(x0, len(x_advs), axis=0) - x_advs
            perturb *= curr_epsilon
            potential_advs = x_advs + perturb
            # Clip (no need for PMU data)
            # potential_advs = np.clip(potential_advs, -3.0, 3.0)
            # preds = model(potential_advs).numpy().argmax(axis=-1)
            preds = model(torch.tensor(potential_advs,
                                       dtype=torch.float32,
                                       device=next(model.parameters()).device)).detach().cpu().numpy().argmax(axis=-1)
            query_cnt += len(x_advs)
            satisfied = (preds != y0)
            epsilon_ratio = np.mean(satisfied)
            # Adjust the epsilon based on the ratio
            if epsilon_ratio < 0.2:
                curr_epsilon *= step_adapt
            elif epsilon_ratio > 0.5:
                curr_epsilon /= step_adapt
            # If found the valid adversarial candidate
            if epsilon_ratio > 0:
                x_adv = _best_adv(x0, potential_advs[np.where(satisfied)[0]])
                break
        else:
            if query_cnt > max_queries:
                print(f"Attack break because more than {max_queries} query!")
                break
            continue
        # Get the current perturbation
        perturbation = x_adv - x0
        perturbation_norm = np.linalg.norm(perturbation)
        # Print the internal results
        if iter % 100 == 0:
            print(f"Current Iteration: {iter}, query number: {query_cnt}.")
            print(f"Current prediction result: {get_probs(model, x_adv, y0)}, perturbation norm: {perturbation_norm}")

        if query_cnt > max_queries:
            print(f"Attack break because more than {max_queries} query!")
            break

        if perturbation_norm < epsilon_l2:
            print("Successfully reached epsilon threshold!")
            print(f"Current Iteration: {iter}, query number: {query_cnt}.")
            print(f"Current prediction result: {get_probs(model, x_adv, y0)}, perturbation norm: {perturbation_norm}")
            break

    perturbation = x_adv - x0
    perturbation_norm = np.linalg.norm(perturbation)
    if perturbation_norm < epsilon_l2:
        return torch.tensor(x_adv), bool(perturbation_norm < epsilon_l2), query_cnt
    else:
        return torch.tensor(x0), bool(perturbation_norm < epsilon_l2), query_cnt
