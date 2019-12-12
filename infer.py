from glob import glob
from cli_utils import get_stuff_from_mode, load_config
from generators import get_gens
import click
from generators import get_gens
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

__all__ = ["infer"]

@click.command("infer")
@click.option("--tub", "-t",  multiple=True)
@click.option("--ckpt", "-c")
@click.option("--config", "-f", type=click.Path(), help="config json file")
def infer(tub, ckpt, config):
    cfg  = load_config(config_json=config)
    if tub:
        tub_paths = tub
    else:
        tub_paths = cfg.get("TUBS", default=())
    print(f"tub_paths -----> {tub_paths}")
    mode = cfg.MODE
    kl, _ = get_stuff_from_mode(mode)
    kl.load_weights(ckpt)

    # smooshing alwasy needs to be True here because we need to is_sim
    # label to match to the embeddings. If False the generator will not
    # return the is_sim labels.
    (_,_), (val_gen, val_count) = get_gens(tub_paths,
                                           batch_size=1,
                                           train_frac=cfg.TRAIN_FRAC,
                                           seed=cfg.RANDOM_SEED,
                                           mode=mode)
    result = {"embedding"     : [],
              "gt_steering"   : [],
              "gt_throttle"   : [],
              "gt_is_sim"     : [],
              "pred_steering" : [],
              "pred_throttle" : []}
    for step in tqdm(range(val_count)):
        x,y = next(val_gen)
        features = kl.get_features(np.array(x[0]))
        steering, throttle = kl.run(x)
        result["embedding"].append(features)
        result["gt_steering"].append(y[0][0])
        result["gt_throttle"].append(y[1][0])
        result["gt_is_sim"].append(y[2][0])
        result["pred_steering"].append(steering)
        result["pred_throttle"].append(throttle)

    ckpt = Path(ckpt)
    json_path = Path(ckpt.parents[0]) / "eval.json"
    with open(json_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    infer()
