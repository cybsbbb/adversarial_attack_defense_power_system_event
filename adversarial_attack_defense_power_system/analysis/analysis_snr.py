from adversarial_attack_defense_power_system.analysis.utils import *


def calculate_snr(x):
    x_signal = spatial_smoothing(x)
    x_noise = x_signal - x
    snr_list = []
    for pmu_idx in range(100):
        # for measurement_idx in range(4):
            signal_seq = x_signal[pmu_idx, :, :, :4]
            noise_seq = x_noise[pmu_idx, :, :, :4]
            # Compute signal power: sum of squares of original data
            signal_power = np.sum(signal_seq ** 2)
            # signal_power = np.std(signal_seq)
            # Compute noise power: sum of squares of perturbation
            noise_power = np.sum(noise_seq ** 2)
            # noise_power = np.std(noise_seq)
            # Compute SNR in dB
            snr_db = 10 * np.log10(signal_power / noise_power)
            snr_list.append(snr_db)
    snr_db_tot = np.mean(snr_list)
    return snr_db_tot, x_noise[:100, :, :, :4]


def stat_snr_changes():
    # Load the data
    original = np.load('../../data/datasets_original/ic_b/test_data.npy')[:, :, :40, :]
    x = np.load('../../data/datasets/ic_b/test_data.npy')[:, :, :40, :]
    x_adv = np.load('../../adv_attack_result/black/b/resnet50/bit_schedule_v6/b_resnet50_bit_schedule_v6/x_adv.npy')
    x_adv = np.transpose(x_adv, (0, 2, 3, 1))
    perturbation = x_adv - x

    # Calculate the norm of the perturbation
    norms = []
    for i in range(perturbation.shape[0]):
        # print(i, np.linalg.norm(perturbation[i].reshape(-1)))
        norms.append(np.linalg.norm(perturbation[i].reshape(-1)))
    norm = np.mean(norms)

    x_denorm = denorm(x, original)
    snr_x, noise1 = calculate_snr(x_denorm)
    print(f"Before attack, snr_db: {snr_x}")

    for perturbation_norm in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        x_adv = x + (perturbation_norm / norm) * perturbation
        x_adv_denorm = denorm(x_adv, original)
        snr_x_adv, noise2 = calculate_snr(x_adv_denorm)
        print(f"After attack: perturbation norm: {perturbation_norm}, SNR diff: {snr_x - snr_x_adv}")
    return


if __name__ == '__main__':
    stat_snr_changes()