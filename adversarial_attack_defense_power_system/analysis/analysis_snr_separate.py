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
            # Compute noise power: sum of squares of perturbation
            noise_power = np.sum(noise_seq ** 2)
            # Compute SNR in dB
            snr_db = 10 * np.log10(signal_power / noise_power)
            snr_list.append(snr_db)
    snr_array = np.array(snr_list)
    std_dev = np.std(snr_array)
    print(std_dev)

    snr_db_tot = np.mean(snr_list)
    return snr_db_tot, x_noise[:100, :, :, :4]


def stat_snr_changes():
    # Load the data
    original = np.load('../../data/datasets_original/ic_b/test_data.npy')[:, :, :40, :]
    x = np.load('../../data/datasets/ic_b/test_data.npy')[:, :, :40, :]
    x_denorm = denorm(x, original)
    snr_x, noise1 = calculate_snr(x_denorm)


if __name__ == '__main__':
    stat_snr_changes()