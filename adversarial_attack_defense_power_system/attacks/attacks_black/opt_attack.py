import torch
import numpy as np
from adversarial_attack_defense_power_system.attacks.attacks_black.utils import get_probs, get_label
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds

"""
This file contains the implementation of the OPT black-box adversarial attack

"""

def distance(x_adv):
    return np.linalg.norm(x_adv)


def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd

    if get_label(model, x0 + lbd * theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd * 1.01
        nquery += 1
        while get_label(model, x0 + lbd_hi * theta) == y0:
            lbd_hi = lbd_hi * 1.01
            nquery += 1
            if lbd_hi > 20:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd * 0.99
        nquery += 1
        while get_label(model, x0 + lbd_lo * theta) != y0:
            lbd_lo = lbd_lo * 0.99
            nquery += 1
            if lbd_lo < 1e-5:
                return float('inf'), nquery

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        nquery += 1
        if get_label(model, x0 + lbd_mid * theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery


def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best:
        if get_label(model, x0 + current_best * theta) == y0:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd

    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        nquery += 1
        if get_label(model, x0 + lbd_mid * theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery


def opt_attack(model, input_image, input_label, attack_config, alpha=0.2, beta=0.005):
    """
        Attack the original image and return adversarial example
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
    print(f"Starting OPT Attack, with max_queries: {max_queries}, epsilon_l2: {epsilon_l2}.")
    setup_random_seeds()
    query_count = 0
    # First random search a fooled sample
    num_directions = 20
    best_theta, g_theta = None, float('inf')
    print("Searching for the initial direction on %d random directions: " % (num_directions))
    for i in range(num_directions):
        query_count += 1
        theta = np.random.randn(*x0.shape) * 10
        if get_label(model, x0 + theta) != y0:
            initial_lbd = np.linalg.norm(theta)
            theta /= initial_lbd
            lbd, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print(f"--------> Found distortion {g_theta:.4f}")
    # First search failed (GG!)
    if g_theta == float('inf'):
        print(f"Failed! Couldn't find valid initial, Total Query Count: {query_count}.")
        return torch.tensor(x0), False, query_count
    print(f"==========> Found best distortion {g_theta:.4f} using {query_count} queries.")

    theta, g2 = best_theta, g_theta
    opt_count = query_count
    for i in range(3000):
        gradient = np.zeros(theta.shape)
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = np.random.randn(*theta.shape)
            u /= np.linalg.norm(u)
            ttt = theta + beta * u
            ttt /= np.linalg.norm(ttt)
            g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd=g2, tol=beta / 500)
            opt_count += count
            gradient += (g1 - g2) / beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0 / q * gradient

        min_theta = theta
        min_g2 = g2

        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta /= np.linalg.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500)
            opt_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                new_theta /= np.linalg.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500)
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta, g2

        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if beta < 1e-8:
                break

        if opt_count > max_queries:
            print(f"Exceed max query number: {max_queries}!")
            break

        dist = distance(g_theta * best_theta)
        if dist * 1.01 < epsilon_l2:
            print("Iteration %3d distortion %.4f num_queries %d" % (i + 1, dist, opt_count))
            break
        print("Iteration %3d distortion %.4f num_queries %d" % (i + 1, dist, opt_count))

    target = get_label(model, x0 + g_theta * best_theta)
    dist = distance(g_theta * best_theta) * 1.01
    print(f"Adversarial Example Found Successfully! Distortion: {dist:.4f}, Target: {target}, Queries: {opt_count}.")

    if dist * 1.01 < epsilon_l2:
        return torch.tensor(x0 + g_theta * best_theta * 1.01), bool(dist < epsilon_l2), opt_count
    else:
        return torch.tensor(x0), bool(dist < epsilon_l2), opt_count
