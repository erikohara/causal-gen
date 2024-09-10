import argparse
import os
import sys
from typing import Dict, Optional, Any

import numpy as np
import pyro
import torch
from torch import nn, Tensor
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/home/erik.ohara/causal-gen/src")
from datasets import cmnist, mimic, morphomnist, ukbb, simba, get_attr_max_min
from vae import HVAE
from hps import Hparams
from utils import EMA, seed_all, seed_worker
from PIL import Image

def vae_preprocess(var_args, pa):
    pa = torch.cat(
        [pa[k] if len(pa[k].shape) > 1 else pa[k][..., None] for k in var_args.parents_x],
        dim=1,
    )
    pa = (
        pa[..., None, None].repeat(1, 1, *(var_args.input_res,) * 2).to(var_args.device).float()
    )
    return pa

def setup_dataloaders(args: Hparams) -> Dict[str, DataLoader]:
    if "ukbb" in args.dataset:
        datasets = ukbb(args)
    elif args.dataset == "morphomnist":
        assert args.input_channels == 1
        assert args.input_res == 32
        assert args.pad == 4
        args.parents_x = ["thickness", "intensity", "digit"]
        args.context_norm = "[-1,1]"
        args.concat_pa = False
        datasets = morphomnist(args)
    elif args.dataset == "cmnist":
        assert args.input_channels == 3
        assert args.input_res == 32
        assert args.pad == 4
        args.parents_x = ["digit", "colour"]
        args.concat_pa = False
        datasets = cmnist(args)
    elif args.dataset == "mimic":
        datasets = mimic(args)
    elif args.dataset == "simba":
        datasets = simba(args)
    else:
        NotImplementedError

    kwargs = {
        "batch_size": args.bs,
        "num_workers": 4,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    dataloaders = {}
    if args.setup == "sup_pgm":
        dataloaders["train"] = DataLoader(
            datasets["train"], shuffle=True, drop_last=True, **kwargs
        )
    else:
        args.n_total = len(datasets["train"])
        args.n_labelled = int(args.sup_frac * args.n_total)
        args.n_unlabelled = args.n_total - args.n_labelled
        idx = np.arange(args.n_total)
        rng = np.random.RandomState(1)
        rng.shuffle(idx)
        train_l = torch.utils.data.Subset(datasets["train"], idx[: args.n_labelled])

        if args.setup == "semi_sup":
            train_u = torch.utils.data.Subset(datasets["train"], idx[args.n_labelled :])
            dataloaders["train_l"] = DataLoader(  # labelled
                train_l, shuffle=True, drop_last=True, **kwargs
            )
            dataloaders["train_u"] = DataLoader(  # unlabelled
                train_u, shuffle=True, drop_last=True, **kwargs
            )
        elif args.setup == "sup_aux":
            dataloaders["train"] = DataLoader(  # labelled
                train_l, shuffle=True, drop_last=True, **kwargs
            )

    dataloaders["valid"] = DataLoader(datasets["valid"], shuffle=False, **kwargs)
    dataloaders["test"] = DataLoader(datasets["test"], shuffle=False, **kwargs)
    return dataloaders

def preprocess(
    batch: Dict[str, Tensor], dataset: str = "ukbb", split: str = "l"
) -> Dict[str, Tensor]:
    if "x" in batch.keys():
        batch["x"] = (batch["x"].float() - 127.5) / 127.5  # [-1,1]
    # for all other variables except x
    not_x = [k for k in batch.keys() if k not in ["x","filename"]]
    for k in not_x:
        if split == "u":  # unlabelled
            batch[k] = None
        elif split == "l":  # labelled
            batch[k] = batch[k].float()
            if len(batch[k].shape) < 2:
                batch[k] = batch[k].unsqueeze(-1)
        else:
            NotImplementedError
    if "ukbb" in dataset:
        for k in not_x:
            if k in ["age", "brain_volume", "ventricle_volume"]:
                k_max, k_min = get_attr_max_min(k)
                batch[k] = (batch[k] - k_min) / (k_max - k_min)  # [0,1]
                batch[k] = 2 * batch[k] - 1  # [-1,1]
    return batch

@torch.no_grad()
def cf_epoch(
    args: Hparams,
    model: nn.Module,
    vae: Optional[nn.Module],
    dataloader: Dict[str, DataLoader],
    do_pa_one: Dict[str, Any],
) -> Dict[str, Any]:
    "supervised epoch"
    loader = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        miniters=len(dataloader) // 100,
        mininterval=5,
    )

    cf_x_all = []
    px_loc_all = []
    cf_pa_all = {}
    do_pa = {}
    filenames = []
    for i, batch in loader:
        bs = batch["x"].shape[0]
        batch = preprocess(batch, args.dataset, split="l")
        pa = {k: v.clone() for k, v in batch.items() if k not in ["x","filename"]}
        for key_do_pa in do_pa_one.keys():
            do_pa[key_do_pa] = torch.tensor([do_pa_one[key_do_pa]]).repeat(bs)[:,None].to(args.device)
        cf_pa = model.counterfactual(
            obs=pa, intervention=do_pa, num_particles=1
        )
        _pa = vae_preprocess(args, {k: v.clone() for k, v in pa.items()})
        _cf_pa = vae_preprocess(args, {k: v.clone() for k, v in cf_pa.items()})
        z_t = 0.1 if "mnist" in args.hps else 1.0
        z = vae.abduct(x=batch["x"], parents=_pa, t=z_t)
        #if vae.cond_prior:
            #z = [z[j]["z"] for j in range(len(z))]
        px_loc, px_scale = vae.forward_latents(latents=z, parents=_pa)
        cf_loc, cf_scale = vae.forward_latents(latents=z, parents=_cf_pa)
        u = (batch["x"] - px_loc) / px_scale.clamp(min=1e-12)
        u_t = 0.1 if "mnist" in args.hps else 1.0  # cf sampling temp
        cf_scale = cf_scale * u_t
        cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1e9, max=1e9)
        #cf_x = cf_loc + cf_scale * u
        cf_x_all.append(cf_x)
        px_loc_all.append(px_loc)
        filenames += batch["filename"]
        if cf_pa_all.keys() != cf_pa.keys():
            cf_pa_all = cf_pa
        else:
            for key_cf_pa in cf_pa.keys():
                cf_pa_all[key_cf_pa] = torch.cat((cf_pa_all[key_cf_pa],cf_pa[key_cf_pa]))
    cf_x_all = torch.cat(cf_x_all)
    #cf_pa_all = torch.cat(cf_pa_all)
    px_loc_all = torch.cat(px_loc_all)
    return {"cf_x": cf_x_all, "rec_x": px_loc_all, "cf_pa": cf_pa_all, "filenames": filenames}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument("--dataset", help="Dataset name.", type=str, default="ukbb")
    parser.add_argument(
        "--data_dir", help="Data directory to load form.", type=str, default=""
    )
    parser.add_argument(
        "--load_path", help="Path to load checkpoint for pgm.", type=str, default=""
    )
    parser.add_argument(
        "--vae_path", help="Path to load checkpoint for vae.", type=str, default=""
    )
    parser.add_argument(
        "--dscm_path", help="Path to load checkpoint for DSCM.", type=str, default=""
    )
    parser.add_argument(
        "--output_path", help="Output path for counterfactuals.", type=str, default=""
    )
    parser.add_argument(
        "--setup",  # semi_sup/sup_pgm/sup_aux
        help="training setup.",
        type=str,
        default="sup_pgm",
    )
    parser.add_argument("--seed", help="Set random seed.", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        help="Toggle cudNN determinism.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--testing", help="Test model.", action="store_true", default=False
    )
    # training
    parser.add_argument(
        "--epochs", help="Number of training epochs.", type=int, default=1000
    )
    parser.add_argument("--bs", help="Batch size.", type=int, default=32)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-4)
    parser.add_argument(
        "--lr_warmup_steps", help="lr warmup steps.", type=int, default=1
    )
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.1)
    parser.add_argument(
        "--input_res", help="Input image crop resolution.", type=int, default=192
    )
    parser.add_argument(
        "--input_channels", help="Input image num channels.", type=int, default=1
    )
    parser.add_argument("--pad", help="Input padding.", type=int, default=9)
    parser.add_argument(
        "--hflip", help="Horizontal flip prob.", type=float, default=0.5
    )
    parser.add_argument(
        "--sup_frac", help="Labelled data fraction.", type=float, default=1
    )
    parser.add_argument("--eval_freq", help="Num epochs per eval.", type=int, default=1)
    # model
    parser.add_argument(
        "--widths",
        help="Cond flow fc network width per layer.",
        nargs="+",
        type=int,
        default=[32, 32],
    )
    parser.add_argument(
        "--parents_x", help="Parents of x to load.", nargs="+", default=[]
    )
    parser.add_argument(
        "--alpha",  # for semi_sup learning only
        help="aux loss multiplier.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--std_fixed", help="Fix aux dist std value (0 is off).", type=float, default=0
    )
    args = parser.parse_known_args()[0]

    seed_all(args.seed, args.deterministic)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    load_path = args.load_path
    vae_path = args.vae_path
    dscm_path = args.dscm_path

    output_path = args.output_path

    # Load data
    dataloaders = setup_dataloaders(args)

    # update hparams if loading checkpoint
    if load_path:
        if os.path.isfile(load_path):
            print(f"\nLoading checkpoint: {load_path}")
            ckpt = torch.load(load_path, map_location=torch.device(device))
            ckpt_args = {k: v for k, v in ckpt["hparams"].items() if k != "load_path"}
            args = Hparams()
            args.update(ckpt_args)
            if args.data_dir is not None:
                ckpt_args["data_dir"] = args.data_dir
            if args.testing:
                ckpt_args["testing"] = args.testing
            vars(args).update(ckpt_args)
        else:
            print(f"Checkpoint not found at: {args.load_path}")

    # Init model
    pyro.clear_param_store()
    if "ukbb" in args.dataset:
        from flow_pgm import FlowPGM

        model = FlowPGM(args)
    elif args.dataset == "morphomnist":
        from flow_pgm import MorphoMNISTPGM

        model = MorphoMNISTPGM(args)
    elif args.dataset == "cmnist":
        from flow_pgm import ColourMNISTPGM

        model = ColourMNISTPGM(args)
    elif args.dataset == "simba":
        from flow_pgm import SimBAPGM

        model = SimBAPGM(args)
    else:
        NotImplementedError
    ema = EMA(model, beta=0.999)
    model.to(device)
    ema.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])

    if vae_path:
        if os.path.isfile(vae_path):
            print(f"\nLoading VAE checkpoint: {vae_path}")
            ckpt_vae = torch.load(vae_path, map_location=torch.device(device))
            ckpt_args_vae = {k: v for k, v in ckpt_vae["hparams"].items() if k != "load_path"}
            args = Hparams()
            args.update(ckpt_args_vae)
        else:
            print(f"Checkpoint for VAE not found at: {vae_path}")
    
    vae = HVAE(args).to(device)

    if dscm_path:
        if os.path.isfile(dscm_path):
            print(f"\nLoading DSCM checkpoint: {dscm_path}")
            dscm_ckpt = torch.load(dscm_path, map_location=torch.device(device))
            ckpt_args_dscm = {k: v for k, v in dscm_ckpt["hparams"].items() if k != "load_path"}
            args = Hparams()
            args.update(ckpt_args_dscm)
        else:
            print(f"Checkpoint for DSCM not found at: {dscm_path}")    

    if dscm_path:
        vae.load_state_dict(
            {
                k[4:]: v
                for k, v in dscm_ckpt["ema_model_state_dict"].items()
                if "vae." in k
            }
        )
    else:
        vae.load_state_dict(ckpt_vae["ema_model_state_dict"])
    ema.ema_model.eval()
    vae.eval()
    
    args = Hparams()
    args.update(ckpt_args)
    args.device = device
    # Most important part of this code
    
    with torch.no_grad():
        do_pa = {'bias_label': 0}

        cf_values = cf_epoch(
            args,
            model=ema.ema_model,
            vae=vae,
            dataloader=dataloaders["test"],
            do_pa_one=do_pa
        )


    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    np_images = cf_values['cf_x'].detach().cpu().numpy()[:,0,:,:]
    print(np_images.shape)
    filenames = cf_values['filenames']

    for index, each_image in enumerate(np_images):
        out_image = Image.fromarray(each_image)
        out_image.save(output_path + f"/{filenames[index].split('.nii')[0]}.tiff")
