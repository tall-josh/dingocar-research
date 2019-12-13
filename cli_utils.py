from collections import namedtuple
import json
from pathlib import Path
import config
from models import  DefaultLinear, CrossentropySmoosh, LinearSmoosh

def get_model(mode, cfg):
    assert mode in list("DLC")
    base_save_dir = getattr(cfg, "BASE_SAVE_DIR", "saved_models")
    if mode == "L":
        model_specific_dir = str(Path(base_save_dir) / 'smoosh_linear')
        numbered_dir = setup_save_dir(model_specific_dir)
        model = LinearSmoosh(save_dir = numbered_dir)
    elif mode == "C":
        model_specific_dir = str(Path(base_save_dir) / 'smoosh_classifier')
        numbered_dir = setup_save_dir(model_specific_dir)
        model = CrossentropySmoosh(save_dir = numbered_dir)
    elif mode == "D":
        model_specific_dir = str(Path(base_save_dir) / 'default')
        numbered_dir = setup_save_dir(model_specific_dir)
        model = DefaultLinear(save_dir = numbered_dir)
    else:
        assert False, "you fuckhead"

    dump_config(mode, cfg, numbered_dir)
    return model

def setup_save_dir(model_specific_dir):
    p = Path(model_specific_dir)
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
        names = [n for n in dir(config) if n[:2] != "__"]
        data  = {n : getattr(config, n) for n in names}

    Config = namedtuple("Config", names)
    class MyConfig(Config):
        def get(self, name, default=None):
            return getattr(self, name, default)
    return MyConfig(**data)

def dump_config(mode, config, save_dir):
    data = config._asdict() # _asdict() returns an OrderedDict
    data["MODE"] = mode
    path = Path(save_dir) / "config.json"
    with path.open('w') as f:
        json.dump(data, f, indent=2)

def dump_history(hist, save_dir):
    result = {}
    for k in hist.history.keys():
        result[k] = [float(n) for n in hist.history[k]]
    
    with (Path(save_dir) / "history.json").open('w') as f:
        json.dump(result, f, indent=2)
