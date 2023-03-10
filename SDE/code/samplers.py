import torch
import numpy as np
import time
import tqdm

def samples_gen(SDENet, f_p, x_init=None, eps=1e-3):
    conf = SDENet.conf
    x_states = []
    if conf.sampler == "PC_origin":
        batch_size = f_p.size(0)
        t = torch.ones(batch_size, device=conf.device)
        if x_init == None:
            x_init = torch.randn(batch_size, SDENet.input_dim, device=conf.device)
        x_init *= SDENet.margin_fn(t)[:, None]
        time_steps = np.linspace(1., eps, SDENet.num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = x_init
        with torch.no_grad():
            for time_step in tqdm.tqdm(time_steps):
                """Corrector Steps"""
                batch_time_step = torch.ones(batch_size, device=conf.device) * time_step
                grad = SDENet(x, f_p, batch_time_step)
                grad_norm = torch.norm(grad, dim=-1).unsqueeze(-1)
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (SDENet.snr * noise_norm / grad_norm)**2
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

                """Predictor Steps"""
                g = SDENet.diffusion_coeff_fn(batch_time_step)
                x_mean = x + (g**2)[:, None] * SDENet(x, f_p, batch_time_step) * step_size
                x_states.append(x_mean)
                x = x_mean + torch.sqrt(g**2 * step_size)[:, None] * torch.randn_like(x)

    elif conf.sampler == "EM":
        batch_size = f_p.size(0)
        t = torch.ones(batch_size, device=conf.device)
        if x_init == None:
            x_init = torch.randn(batch_size, SDENet.input_dim, device=conf.device)
        x_init *= SDENet.margin_fn(t)[:, None]
        time_steps = np.linspace(1., eps, SDENet.num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = x_init
        with torch.no_grad():
            for time_step in tqdm.tqdm(time_steps):
                """Predictor Steps"""
                batch_time_step = torch.ones(batch_size, device=conf.device) * time_step
                g = SDENet.diffusion_coeff_fn(batch_time_step)
                x_mean = x + (g**2)[:, None] * SDENet(x, f_p, batch_time_step) * step_size
                x_states.append(x_mean)
                x = x_mean + torch.sqrt(g**2 * step_size)[:, None] * torch.randn_like(x)

    return {"x_states": x_states}
