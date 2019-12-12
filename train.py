from glob import glob
import json
from pathlib import Path
from generators import get_gens
from cli_utils import (get_stuff_from_mode, load_config, setup_outdir,
                       dump_config, dump_history)
import click

__all__ = ["train"]

@click.command("train")
@click.option("--tub", "-t", type=click.Path(), multiple=True)
@click.option("--mode", "-m", type=str, help="(D)efault | (L)inear_smoosh | (C)ategorical_smoosh")
def train(tub, mode):
    cfg = load_config()

    if tub is not ():
        tub_paths = tub
    else:
        tub_paths = cfg.get("TUBS", default=())

    kl, saved_model_dir = get_stuff_from_mode(mode)
    (train_gen, train_steps), (val_gen, val_steps) = get_gens(
                                                        tub_paths,
                                                        batch_size=cfg.BATCH_SIZE,
                                                        train_frac=cfg.TRAIN_FRAC,
                                                        seed=cfg.RANDOM_SEED,
                                                        mode = mode
                                                        )

    outdir = setup_outdir(saved_model_dir, cfg)
    dump_config(mode, cfg, outdir)
    saved_model_path = str(Path(outdir, "saved_weights.h5"))
    hist = kl.train(train_gen, val_gen, train_steps, val_steps,
             outdir,
             epochs=cfg.EPOCHS,
             verbose=1,
             min_delta=.0005,
             use_early_stop=cfg.USE_EARLY_STOP,
             save_nth=cfg.SAVE_NTH,
             patience=cfg.EARLY_STOP_PATIENTCE
             )

    #import pdb; pdb.set_trace()
    dump_history(hist, outdir)

if __name__ == "__main__":
    train()

