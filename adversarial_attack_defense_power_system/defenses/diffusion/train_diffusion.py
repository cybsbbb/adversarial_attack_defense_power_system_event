from adversarial_attack_defense_power_system.defenses.diffusion.gaussian_diffusion import *
from adversarial_attack_defense_power_system.defenses.diffusion.unet_2d import *


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    for interconnection in ['b', 'c'][:]:
        for timesteps in [20, 50, 100, 300, 1000][:]:
            for beta_type in ['linear', 'cosine'][:]:
                model = Unet()
                diffusion_model = GaussianDiffusion(model=model,
                                                    interconnection=interconnection,
                                                    timesteps=timesteps,
                                                    beta_type=beta_type)
                diffusion_model.to_device(device)
                diffusion_model.train_diffusion(epoch=500)
