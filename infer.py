from glob import glob
from experimental_models import KerasLinearAdversarialDistributionSmoosher
from generators import get_gens
import click
from generators import get_gens
import json
from tqdm import tqdm

__all__ = ["infer"]

@click.command("infer")
@click.option("--tub", "-t", type=click.Path(), multiple=True)
@click.option("--outdir", "-o", type=click.Path())
@click.option("--ckpt", "-c", type=click.Path())
@click.option("--mode", "-m", type=str)
def infer(tub, outdir, ckpt, mode):
    tub_paths = tub
    if mode == "smoosh":
        from smoosh_model import default_n_linear
        saved_model_dir = "smoosh_models"
    elif mode == "linear":
        from default_model import default_n_linear
        saved_model_dir = "dual_data_models"
    else:
        print("you fuckhead")
        return -1

    model = default_n_linear()
    kl = KerasLinearAdversarialDistributionSmoosher(model=model)
    kl.compile()
    kl.load_weights(ckpt)

    # smooshing alwasy needs to be True here because we need to is_sim
    # label to match to the embeddings. If False the generator will not
    # return the is_sim labels.
    (_,_), (val_gen, val_count) = get_gens(tub_paths, batch_size=1, smooshing = True)

    result = {"embedding" : [], "is_sim" : []}
    for step in tqdm(range(val_count)):
        x,y = next(val_gen)
        features = kl.get_features(x)
        result["embedding"].append(features[0].tolist())
        result["is_sim"].append(y[2][0])

    with open("result.json", 'w') as f:
        json.dump(result, f)



if __name__ == "__main__":
    train()
