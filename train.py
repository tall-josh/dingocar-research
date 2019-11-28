from glob import glob
from experimental_models import KerasLinearAdversarialDistributionSmoosher
from network_defs import default_n_linear, smoosh_classification, smoosh_linear
from generators import get_gens
import click

__all__ = ["train"]

@click.command("train")
@click.option("--tub", "-t", type=click.Path(), multiple=True)
@click.option("--outdir", "-o", type=click.Path())
@click.option("--model", "-m", type=str, help="(D)fault | (L)inear | (C)lassification")
def train(tub, outdir, mode):
    saved_model_dir = outdir
    tub_paths = tub
    assert model in list("DLC")

    if model == "L":
        model = smoosh_linear()
        saved_model_dir = "saved_models/smoosh_linear
        smooshing = true
    elif model == "C":
        model = smoosh_linear()
        saved_model_dir = "saved_models/smoosh_classifier
        smooshing = true
    elif model == "D":
        model = default_n_linear()
        saved_model_dir = "saved_models/default"
        smooshing = False
    else:
        print("you fuckhead")
        return -1

    kl = KerasLinearAdversarialDistributionSmoosher(model=model)
    kl.compile()

    (train_gen, train_steps), (val_gen, val_steps) = get_gens(tub_paths, smooshing=smooshing)

    num = len(glob(f"{saved_model_dir}/*.h5"))
    saved_model_path = f"{saved_model_dir}/{num:02d}.h5"
    kl.train(train_gen, val_gen, train_steps, val_steps,
             saved_model_path, epochs=50,
                  verbose=1, min_delta=.0005, patience=5, use_early_stop=True)


if __name__ == "__main__":
    train()
