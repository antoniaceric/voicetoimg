import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pathlib import Path  # ← make sure this is at the top

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# Set seed for reproducibility
import random
SEED = 13 # ← change to any constant
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # ------------- device handling -------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # ► the CLIP text-encoder lives in model.cond_stage_model
    enc = model.cond_stage_model
    enc.to(device)
    enc.device = device.type         # overwrite the cached attribute the class sets in __init__
    # -------------------------------------------
    print('**********CPU activated*****', device)

    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="tree on grass",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
    "--init_latents",
    type=str,
    default="",
    help="Path to .pt tensor (N,C,H,W). If given, used as x_T instead of random noise."
    )
    
    parser.add_argument(
        "--subject_idx", type=int, default=0,
        help="Index of the subject in the latent tensor batch"
    )

    opt = parser.parse_args()


    config = OmegaConf.load("/files/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [4, opt.H // 8, opt.W // 8]

                # ---------- optional initial latents ----------
                x_T = None
                if opt.init_latents:
                    print("Loading initial latents:", opt.init_latents)
                    all_latents = torch.load(opt.init_latents, map_location=device)

                    if opt.subject_idx >= all_latents.shape[0]:
                        raise ValueError(f"subject_idx {opt.subject_idx} out of range (max={all_latents.shape[0]-1})")

                    x_T = all_latents[opt.subject_idx].unsqueeze(0).to(device)
                    print(f"  Loaded subject {opt.subject_idx} → shape {x_T.shape}")
                # ----------------------------------------------

                samples, _ = sampler.sample(
                    S=opt.ddim_steps,
                    conditioning=c,
                    batch_size=x_T.size(0) if x_T is not None else opt.n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    x_T=x_T
                )

                x_samples_ddim = model.decode_first_stage(samples)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    filename = f"subject_{opt.subject_idx:03}_sample_{base_count + i:04}.png"
                    Image.fromarray(x_sample.astype(np.uint8)).save(Path(sample_path) / filename)

                all_samples.append(x_samples_ddim)



    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
