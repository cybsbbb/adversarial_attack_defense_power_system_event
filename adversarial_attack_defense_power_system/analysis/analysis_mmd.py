from adversarial_attack_defense_power_system.analysis.utils import *
from adversarial_attack_defense_power_system.analysis.analysis_snr import *


def stat_mmd_changes():
    # Load the data
    original = np.load('../../data/datasets_original/ic_b/test_data.npy')[:, :, :40, :]
    x = np.load('../../data/datasets/ic_b/test_data.npy')[:, :, :40, :]
    x_adv = np.load('../../adv_exp_result/black/max_queries_1000_epsilon_l2_40'
                    '/b/resnet50/bit_schedule_v6/b_resnet50_bit_schedule_v6/x_adv.npy')
    x_adv = np.transpose(x_adv, (0, 2, 3, 1))
    perturbation = x_adv - x

    # Calculate the norm of the perturbation
    norms = []
    for i in range(perturbation.shape[0]):
        norms.append(np.linalg.norm(perturbation[i].reshape(-1)))
    norm = np.mean(norms)

    # extract original sample's noise
    x_denorm = denorm(x, original)
    _, noise1 = calculate_snr(x_denorm)

    for perturbation_norm in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        # extract attacked sample's noise
        x_adv = x + (perturbation_norm / norm) * perturbation
        x_adv_denorm = denorm(x_adv, original)
        _, noise2 = calculate_snr(x_adv_denorm)

        dataA = noise1
        dataB = noise2

        def rbf_kernel(X, Y, gamma=0.1):
            """
            计算 RBF 核矩阵：K(x, y) = exp(-gamma * ||x - y||^2)
            X: (n_x, d)
            Y: (n_y, d)
            """
            X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)  # (n_x, 1)
            Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)  # (1, n_y)
            dist_sq = X_sq + Y_sq - 2 * np.dot(X, Y.T)
            K = np.exp(-gamma * dist_sq)
            return K

        def mmd_sq(X, Y, gamma=0.1):
            """
            计算 MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
            X, Y: shape = (n_samples, n_features)
            """
            Kxx = rbf_kernel(X, X, gamma=gamma)
            Kyy = rbf_kernel(Y, Y, gamma=gamma)
            Kxy = rbf_kernel(X, Y, gamma=gamma)

            mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
            return mmd2

        # 假设 dataA, dataB 的形状均为 (500, 40, 360)
        # 1) Flatten
        dataA_flat = dataA.reshape(dataA.shape[0], -1)  # (500, 14400)
        dataB_flat = dataB.reshape(dataB.shape[0], -1)  # (500, 14400)

        # 2) 计算 MMD^2
        mmd2_value = mmd_sq(dataA_flat, dataB_flat)  # gamma根据数据尺度调参
        # print("MMD^2 =", mmd2_value)

        # 3) 置换检验 (permutation test)
        #    原理：将 A、B 合并后随机打乱分配为 A、B，看在置换情况下MMD^2的分布。
        all_data_flat = np.vstack([dataA_flat, dataB_flat])  # (1000, 14400)
        labels = np.array([0] * dataA.shape[0] + [1] * dataB.shape[0])
        n_permutations = 1000  # 可根据需求增大
        perm_mmd2 = []
        for i in range(n_permutations):
            np.random.shuffle(labels)
            X_perm = all_data_flat[labels == 0]
            Y_perm = all_data_flat[labels == 1]
            perm_mmd2.append(mmd_sq(X_perm, Y_perm))

        perm_mmd2 = np.array(perm_mmd2)
        # 计算 p-value: 看原始 mmd2_value 在置换分布中多“极端”
        p_value = np.mean(perm_mmd2 > mmd2_value)
        print(f"perturbation_norm: {perturbation_norm}, Permutation test p-value = {p_value}")
    return


if __name__ == '__main__':
    stat_mmd_changes()
