import torch
import numpy as np
from adversarial_attack_defense_power_system.attacks.attacks_black.utils import get_probs, get_label
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds

"""
This file contains the implementation of the Sign-OPT black-box adversarial attack

"""

def distance(x_adv):
    return np.linalg.norm(x_adv)


def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign == 0] = 1
    return y_sign


def sign_grad_v1(model, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
    """
    Evaluate the sign of gradient by formulat
    sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
    """
    K = 200
    sign_grad = np.zeros_like(theta)
    queries = 0
    for _ in range(K):
        u = np.random.randn(*theta.shape)
        u /= np.linalg.norm(u)

        sign = 1
        new_theta = theta + h * u
        new_theta /= np.linalg.norm(new_theta)

        # Targeted case.
        if (target is not None and
                get_label(model, x0 + initial_lbd * new_theta) == target):
            sign = -1

        # Untargeted case
        if (target is None and
                get_label(model, x0 + initial_lbd * new_theta) != y0):
            sign = -1
        queries += 1
        sign_grad += u * sign

    sign_grad /= K

    return sign_grad, queries


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
            if lbd_hi > 100:
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


def sign_opt_attack(model, input_image, input_label, attack_config, alpha=0.2, beta=0.005, momentum=0.0):
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
    print(f"Starting Sign-OPT Attack, with max_queries: {max_queries}, epsilon_l2: {epsilon_l2}.")
    setup_random_seeds()
    query_count = 0
    # Calculate a good starting point.
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

    # Begin Gradient Descent.
    xg, gg = best_theta, g_theta
    vg = np.zeros_like(xg)

    for i in range(3000):
        sign_gradient, grad_queries = sign_grad_v1(model, x0, y0, xg, initial_lbd=gg, h=beta)

        query_count += grad_queries
        # Line search
        min_theta = xg
        min_g2 = gg
        min_vg = vg
        for _ in range(15):
            if momentum > 0:
                new_vg = momentum * vg - alpha * sign_gradient
                new_theta = xg + new_vg
            else:
                new_theta = xg - alpha * sign_gradient
            new_theta /= np.linalg.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd=min_g2,
                                                             tol=beta / 500)
            query_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
                if momentum > 0:
                    min_vg = new_vg
            else:
                break
        if min_g2 >= gg:
            for _ in range(15):
                alpha = alpha * 0.25
                if momentum > 0:
                    new_vg = momentum * vg - alpha * sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= np.linalg.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd=min_g2,
                                                                 tol=beta / 500)
                query_count += count
                if new_g2 < gg:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if momentum > 0:
                        min_vg = new_vg
                    break
        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving")
            beta = beta * 0.1
            if (beta < 1e-8):
                break

        xg, gg = min_theta, min_g2
        vg = min_vg

        if query_count > max_queries:
            break

        dist = distance(gg * xg)
        if dist * 1.01 < epsilon_l2:
            print("Iteration %3d distortion %.4f num_queries %d" % (i + 1, dist, query_count))
            break
        print("Iteration %3d distortion %.4f num_queries %d" % (i + 1, dist, query_count))

    target = get_label(model, x0 + gg * xg)
    dist = distance(gg * xg) * 1.01
    print(f"Adversarial Example Found Successfully! Distortion: {dist:.4f}, Target: {target}, Queries: {query_count}.")

    if dist * 1.01 < epsilon_l2:
        return torch.tensor(x0 + gg * xg * 1.01), bool(dist < epsilon_l2), query_count
    else:
        return torch.tensor(x0), bool(dist < epsilon_l2), query_count
