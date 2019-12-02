from glob import glob
from generators import get_gens
from cli_utils import get_stuff_from_mode
import click

__all__ = ["train"]

@click.command("train")
@click.option("--tub", "-t", type=click.Path(), multiple=True)
@click.option("--mode", "-m", type=str, help="(D)efault | (L)inear_smoosh | (C)ategorical_smoosh")
def train(tub, mode):
    #saved_model_dir = outdir
    tub_paths = tub
    kl, saved_model_dir = get_stuff_from_mode(mode)
    (train_gen, train_steps), (val_gen, val_steps) = get_gens(tub_paths, mode = mode)

    num = len(glob(f"{saved_model_dir}/*.h5"))
    saved_model_path = f"{saved_model_dir}/{num:02d}.h5"
    kl.train(train_gen, val_gen, train_steps, val_steps,
             saved_model_path, epochs=50,
             verbose=1, min_delta=.0005, patience=5)


if __name__ == "__main__":
    train()
