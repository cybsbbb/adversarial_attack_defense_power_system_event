import os
import math
import numpy as np
import torch.cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from adversarial_attack_defense_power_system.dataset_loader.dataset_loader import load_dataset_npy
from adversarial_attack_defense_power_system.defenses.diffusion.beta_schedule import *
from adversarial_attack_defense_power_system.defenses.diffusion.helper_functions import *
from adversarial_attack_defense_power_system.utils.random_seeds_setups import setup_random_seeds


class GaussianDiffusion:
    def __init__(
            self,
            model,
            interconnection,
            *,
            loss_type='huber',
            timesteps=1000,
            objective='pred_noise',
            beta_type='linear',
    ):
        super().__init__()

        self.model = model
        self.interconnection = interconnection
        if interconnection == 'b' or interconnection == 'B':
            self.sample_size = (360, 40, 4)
        elif interconnection == 'c' or interconnection == 'C':
            self.sample_size = (360, 176, 4)
        else:
            ValueError(f'invalid interconnection {self.interconnection}')
        self.channels = self.model.channels
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.objective = objective
        self.beta_type = beta_type

        # define beta type
        self.betas = get_beta_schedule(timesteps=timesteps, beta_type=beta_type)

        # define the alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    # Load Weights
    def load(self):
        script_path = Path(__file__).resolve().parent
        if self.model.model_name == 'unet_1d':
            weights_path = f'{script_path}/../../../weights/diffusion/ic_{self.interconnection}/weights/' \
                           f'ic_{self.interconnection}_t{self.timesteps}_{self.beta_type}_1d.pth'
        elif self.model.model_name == 'unet_2d':
            weights_path = f'{script_path}/../../../weights/diffusion/ic_{self.interconnection}/weights/' \
                           f'ic_{self.interconnection}_t{self.timesteps}_{self.beta_type}.pth'
        else:
            raise ValueError(f"Invalid model_name: {self.model.model_name}")
        print(f"Loading model weights from: {weights_path}.")
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        return 0

    # Save Weights
    def save(self, weights_path):
        print(f"Saving model weights to: {weights_path}.")
        torch.save(self.model.state_dict(), weights_path)
        return 0

    # Move model to device
    def to_device(self, device):
        print(f"Moving model to: {device}.")
        self.model.to(device)
        return 0

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Loss Function
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    # Forward Loss
    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        loss = self.loss_fn(noise, predicted_noise)
        return loss

    # Single-step backward process (DDPM)
    @torch.no_grad()
    def p_sample(self, x, t, t_index, noise=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            if noise == None:
                noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Transformation ddpm
    def transformation_sample_ddpm(self, event_sample, steps=2):
        # Forward process to add noise to the sample
        device = next(self.model.parameters()).device
        b = event_sample.shape[0]
        t = torch.full((b,), steps-1, device=device, dtype=torch.long)
        noise_sample = self.q_sample(event_sample, t)
        for t in range(steps)[::-1]:
            b = torch.tensor([t], device=device)
            noise_sample = self.p_sample(noise_sample, b, t)
        return noise_sample

    # Transformation ddpm (Return the whole intervals)
    def transformation_sample_intervals_ddpm(self, event_sample, steps=2):
        # Forward process to add noise to the sample
        device = next(self.model.parameters()).device
        b = event_sample.shape[0]
        t = torch.full((b,), steps-1, device=device, dtype=torch.long)
        noise_sample = self.q_sample(event_sample, t)
        intervals = []
        intervals.append(torch.clone(noise_sample))
        for t in range(steps)[::-1]:
            b = torch.tensor([t], device=device)
            noise_sample = self.p_sample(noise_sample, b, t)
            intervals.append(torch.clone(noise_sample))
        return intervals

    # Transformation ddim
    # Diffusion_t: forward process deep between 0 and 1
    # denoise_steps: Total step for de-noising
    @torch.no_grad()
    def transformation_sample_ddim(self, event_sample, *, diffusion_t=0.1, denoise_steps=2, ddim_eta=0):
        ddim_timesteps = int(denoise_steps / diffusion_t)
        if ddim_timesteps > self.timesteps:
            ddim_timesteps = self.timesteps
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(self.model.parameters()).device
        b = event_sample.shape[0]
        t = torch.full((b,), denoise_steps * c - 1, device=device, dtype=torch.long)
        noise_sample = self.q_sample(event_sample, t)

        for i in reversed(range(0, denoise_steps + 1)):
            t = torch.full((b,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((b,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, t, noise_sample.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, noise_sample.shape)
            # 2. predict noise using model
            pred_noise = self.model(noise_sample, t)
            # 3. get the predicted x_0
            pred_x0 = (noise_sample - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(noise_sample)

            noise_sample = x_prev
        return noise_sample

    @torch.no_grad()
    def transformation_sample_intervals_ddim(self, event_sample, *, diffusion_t=0.1, denoise_steps=2, ddim_eta=0):
        ddim_timesteps = int(denoise_steps / diffusion_t)
        if ddim_timesteps > self.timesteps:
            ddim_timesteps = self.timesteps
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(self.model.parameters()).device
        b = event_sample.shape[0]
        t = torch.full((b,), denoise_steps * c - 1, device=device, dtype=torch.long)
        noise_sample = self.q_sample(event_sample, t)
        intervals = []
        intervals.append(torch.clone(noise_sample))
        for i in reversed(range(0, denoise_steps + 1)):
            t = torch.full((b,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((b,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, t, noise_sample.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, noise_sample.shape)
            # 2. predict noise using model
            pred_noise = self.model(noise_sample, t)
            # 3. get the predicted x_0
            pred_x0 = (noise_sample - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(
                noise_sample)
            noise_sample = x_prev
            intervals.append(torch.clone(noise_sample))
        return intervals

    # Transformation for dataset
    def transformation_dataset(self, data, parameters, transformation_type='ddpm'):
        # Get model's device
        device = next(self.model.parameters()).device

        data_res = np.zeros_like(data)
        data = torch.tensor(np.transpose(data, (0, 3, 1, 2)), device=device, dtype=torch.float32)

        # for idx in range(data.shape[0]):
        for idx in tqdm(range(data.shape[0])):
            event_sample = data[idx:idx + 1]
            if transformation_type == 'ddpm':
                noise_sample = self.transformation_sample_ddpm(event_sample, **parameters)
            elif transformation_type == 'ddim':
                noise_sample = self.transformation_sample_ddim(event_sample, **parameters)
            else:
                raise ValueError(f'invalid transformation type {transformation_type}')
            data_res[idx:idx + 1] = np.transpose(noise_sample.cpu().numpy(), (0, 2, 3, 1))
        return data_res

    # Train the diffusion model
    def train_diffusion(self, epoch=500):
        dataset = load_dataset_npy(interconnection=self.interconnection)
        train_data = dataset['train_data']
        print(f"Training Diffusion for interconnection: {self.interconnection}, "
              f"timesteps: {self.timesteps}, beta_type: {self.beta_type}")

        # Create Training DataLoader
        X_train = torch.tensor(np.transpose(train_data, (0, 3, 1, 2))).float()
        print(X_train.shape)
        dataloader = DataLoader(X_train, batch_size=16, shuffle=True)

        device = next(self.model.parameters()).device

        # Create dir for weights and losses
        script_path = Path(__file__).resolve().parent
        weights_dir = f'{script_path}/../../../weights/diffusion/ic_{self.interconnection}/weights'
        losses_dir = f'{script_path}/../../../weights/diffusion/ic_{self.interconnection}/losses'
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        if not os.path.exists(losses_dir):
            os.makedirs(losses_dir)
        if self.model.model_name == 'unet_1d':
            weights_path = f'{weights_dir}/ic_{self.interconnection}_t{self.timesteps}_{self.beta_type}_1d.pth'
            losses_path = f'{losses_dir}/ic_{self.interconnection}_t{self.timesteps}_{self.beta_type}_1d.png'
        elif self.model.model_name == 'unet_2d':
            weights_path = f'{weights_dir}/ic_{self.interconnection}_t{self.timesteps}_{self.beta_type}.pth'
            losses_path = f'{losses_dir}/ic_{self.interconnection}_t{self.timesteps}_{self.beta_type}.png'
        else:
            raise ValueError(f"Invalid model_name: {self.model.model_name}")

        # Define the optimizer
        optimizer = Adam(self.model.parameters(), lr=5e-4)
        epochs = epoch

        losses = []
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                batch_size = batch.shape[0]
                batch = batch.to(device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

                loss = self.p_losses(batch, t)

                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().numpy()))
        plt.plot(losses)
        plt.savefig(losses_path)
        plt.close()

        # Save the weights and clean the GPU
        self.save(weights_path)
        torch.cuda.empty_cache()

        return 0
