from adversarial_attack_defense_power_system.defenses.input_transformation.spatial_smoothing import spatial_smoothing
from adversarial_attack_defense_power_system.defenses.input_transformation.low_pass_filtering import low_pass_filtering
from adversarial_attack_defense_power_system.defenses.input_transformation.feature_squeezing import feature_squeezing
from adversarial_attack_defense_power_system.defenses.input_transformation.svd_decomposition import svd_decomposition
from adversarial_attack_defense_power_system.defenses.input_transformation.event_decomposition import event_decomposition
# from adversarial_attack_defense_power_system.defenses.encoders.encoders_wrapper import encoders_wrapper
from adversarial_attack_defense_power_system.defenses.diffusion.diffusion_tranformation_wrapper import diffusion_dataset_transformation


def input_transformation_wrapper(data, type="spatial_smoothing", parameters=None):
    if type == 'spatial_smoothing':
        return spatial_smoothing(data)
    elif type == 'low_pass_filtering':
        return low_pass_filtering(data)
    elif type == 'feature_squeezing':
        return feature_squeezing(data)
    elif type == 'svd_decomposition':
        return svd_decomposition(data)
    elif type == 'event_decomposition':
        return event_decomposition(data)
    # elif type == 'autoencoder':
    #     return encoders_wrapper(data, type)
    # elif type == 'cnn_1d':
    #     return encoders_wrapper(data, type)
    # elif type == 'cnn_2d':
    #     return encoders_wrapper(data, type)
    elif type == 'diffusion':
        return diffusion_dataset_transformation(data, parameters)
    elif type == '':
        return data
    else:
        raise ValueError("Wrong preprocessing type!")
