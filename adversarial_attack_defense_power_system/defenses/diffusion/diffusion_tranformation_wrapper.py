import time
from adversarial_attack_defense_power_system.defenses.diffusion.gaussian_diffusion import *
from adversarial_attack_defense_power_system.defenses.diffusion.unet_2d import *


# Setup the device
if torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using Device: {device}")


def diffusion_dataset_transformation(dataset, diffusion_parameters):
    # Load the diffusion model
    interconnection = diffusion_parameters['interconnection']
    timesteps = diffusion_parameters['timesteps']
    beta_type = diffusion_parameters['beta_type']
    model = Unet()
    diffusion_model = GaussianDiffusion(model=model, interconnection=interconnection, timesteps=timesteps, beta_type=beta_type)
    diffusion_model.load()
    diffusion_model.to_device(device)
    # Input Transformation depend on the schedular
    transformation_type = diffusion_parameters['transformation_type']
    if transformation_type == 'ddpm':
        steps = diffusion_parameters['steps']
        parameter = {'steps': steps}
    elif transformation_type == 'ddim':
        diffusion_t = diffusion_parameters['diffusion_t']
        denoise_steps = diffusion_parameters['denoise_steps']
        parameter = {'diffusion_t': diffusion_t, 'denoise_steps': denoise_steps, 'ddim_eta': 0}
    else:
        raise ValueError(f'invalid transformation type {transformation_type}')
    dataset_transformed = diffusion_model.transformation_dataset(dataset, parameters=parameter, transformation_type=transformation_type)
    return dataset_transformed


if __name__ == '__main__':
    timesteps = 20
    beta_type = 'linear'
    interconnection = 'b'
    network_type = 'InfoLoad-l2-regularization'
    model_name = f'ic_{interconnection}_{network_type}'
    ddim_timesteps = 10
    diffusion_t = 0.1
    ddim_parameter = {'interconnection': interconnection,
                      'timesteps': timesteps,
                      'beta_type': beta_type,
                      'transformation_type': 'ddim',
                      'diffusion_t': diffusion_t,
                      'denoise_steps': 3}
    print(ddim_parameter)

    attack_type = 'fgsm'
    test_data_adv = np.load(f'../../../data/adversarial_testing_datasets/{model_name}/{attack_type}/test_data.npy')
    print(test_data_adv)
    test_data_adv_cleaned = diffusion_dataset_transformation(test_data_adv, ddim_parameter)
    print(test_data_adv_cleaned.shape)
