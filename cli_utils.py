from collections import namedtuple
import json
from pathlib import Path
import config as cfg

def get_stuff_from_mode(mode):
    assert mode in list("DLC")
    save_dir = getattr(cfg, "SAVE_DIR", "saved_models")
    if mode == "L":
        from models import LinearSmoosh
        from layers import smoosh_linear
        from losses import is_sim_linear_loss
        kl = LinearSmoosh(model = smoosh_linear())
        loss = is_sim_linear_loss()
        saved_model_dir = f"{save_dir}/smoosh_linear"
    elif mode == "C":
        from models import CrossentropySmoosh
        from layers import smoosh_classification
        from losses import is_sim_categorical_loss
        kl = CrossentropySmoosh( model = smoosh_classification())
        loss = is_sim_categorical_loss()
        saved_model_dir = f"{save_dir}/smoosh_classifier"
    elif mode == "D":
        from models import KerasLinear
        from layers import default_n_linear
        kl = KerasLinear(model = default_n_linear())
        loss = "mse"
        saved_model_dir = f"{save_dir}/default"
    else:
        assert False, "you fuckhead"

    kl.compile(loss=loss)
    return kl, saved_model_dir

def setup_outdir(saved_model_dir, cfg):
    p = Path(saved_model_dir)
    num = len([x for x in p.iterdir() if x.is_dir()])
    p = p/f"{num:02}"
    p.mkdir()
    return str(p)

def load_config(config_json=None):
    if config_json is not None:
        with open(config_json, 'r') as f:
            data = json.load(f)
        names = list(data.keys())
    else:
        names = [n for n in dir(cfg) if n[:2] != "__"]
        data  = {n : getattr(cfg, n) for n in names}

    Config = namedtuple("Config", names)
    class MyConfig(Config):
        def get(self, name, default=None):
            return getattr(self, name, default)
    return MyConfig(**data)

def dump_config(mode, config, outdir):
    data = config._asdict() # _asdict() returns an OrderedDict
    data["MODE"] = mode
    path = Path(outdir) / "config.json"
    with path.open('w') as f:
        json.dump(data, f, indent=2)

def dump_history(hist, outdir):
    result = {}
    for k in hist.history.keys():
        result[k] = [float(n) for n in hist.history[k]]
    
    with (Path(outdir) / "history.json").open('w') as f:
        json.dump(result, f, indent=2)
