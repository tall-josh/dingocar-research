from glob import glob
from models import  KerasLinear, CrossentropySmoosh, LinearSmoosh
from generators import get_gens
import click
from generators import get_gens
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

__all__ = ["infer"]

@click.command("infer")
@click.option("--tub", "-t", type=click.Path(), multiple=True)
@click.option("--ckpt", "-c", type=click.Path())
@click.option("--model", "-m", type=str)
def infer(tub, ckpt, model):
    tub_paths = tub
    if model == "L":
        smooshing = True
        kl = LinearSmoosh()
    elif model == "C":
        smooshing = True
        kl = CrossentropySmoosh()
    elif model == "D":
        smooshing = False
        kl = KerasLinear()
    else:
        print("you fuckhead")
        return -1

    kl.compile()
    kl.load_weights(ckpt)

    # smooshing alwasy needs to be True here because we need to is_sim
    # label to match to the embeddings. If False the generator will not
    # return the is_sim labels.
    (_,_), (val_gen, val_count) = get_gens(tub_paths, batch_size=1, smooshing = True)

    result = {"embedding"     : [],
              "gt_steering"   : [],
              "gt_throttle"   : [],
              "gt_is_sim"     : [],
              "pred_steering" : [],
              "pred_throttle" : []}
    for step in tqdm(range(val_count)):
        x,y = next(val_gen)
        features = kl.get_features(np.array(x))
        steering, throttle = kl.run(x)
        result["embedding"].append(features)
        result["gt_steering"].append(y[0][0])
        result["gt_throttle"].append(y[1][0])
        result["gt_is_sim"].append(y[2][0])
        result["pred_steering"].append(steering)
        result["pred_throttle"].append(throttle)

    ckpt = Path(ckpt)
    saved_model_dir = ckpt.parents[0]
    json_path = (saved_model_dir / ckpt.stem).with_suffix(".json")
    with open(json_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    train()
