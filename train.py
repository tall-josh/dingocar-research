from glob import glob
from models import  KerasLinear, CrossentropySmoosh, LinearSmoosh
from generators import get_gens
import click

__all__ = ["train"]

@click.command("train")
@click.option("--tub", "-t", type=click.Path(), multiple=True)
@click.option("--outdir", "-o", type=click.Path())
@click.option("--model", "-m", type=str, help="(D)fault | (L)inear | (C)lassification")
def train(tub, outdir, model):
    saved_model_dir = outdir
    tub_paths = tub
    assert model in list("DLC")

    if model == "L":
        saved_model_dir = "saved_models/smoosh_linear"
        smooshing = True
        kl = LinearSmoosh()
    elif model == "C":
        saved_model_dir = "saved_models/smoosh_classifier"
        smooshing = True
        kl = CrossentropySmoosh()
    elif model == "D":
        saved_model_dir = "saved_models/default"
        smooshing = False
        kl = KerasLinear()
    else:
        print("you fuckhead")
        return -1

    kl.compile()

    (train_gen, train_steps), (val_gen, val_steps) = get_gens(tub_paths, smooshing=smooshing)

    num = len(glob(f"{saved_model_dir}/*.h5"))
    saved_model_path = f"{saved_model_dir}/{num:02d}.h5"
    kl.train(train_gen, val_gen, train_steps, val_steps,
             saved_model_path, epochs=50,
             verbose=1, min_delta=.0005, patience=5)


if __name__ == "__main__":
    train()
