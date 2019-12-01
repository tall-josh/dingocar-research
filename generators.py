import numpy as np
import json
from PIL import Image
from pathlib import Path

'''record_###.json
{"user/throttle": 0.7424248049084544, "timestamp": "2019-03-06 21:04:48.552623", "user/mode": "user", "cam/image_array": "219_cam-image_array_.jpg", "user/angle": 0.12798827723777417, "is-simulated" : false}
'''
THROTTLE="user/throttle"
STEERING="user/angle"
IS_SIM="is-simulated"
IMAGE="cam/image_array"

def norm_image(image):
    return (image / 255.) - 0.5

def load_image(img):
    img = np.asarray(Image.open(img), dtype=np.float32)
    img = norm_image(img)
    return img

def batcher(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_gen(data, batch_size, mode):
    while True:
        np.random.shuffle(data)
        for batch in batcher(data, batch_size):
            x = []
            steering = []
            throttle = []
            is_sim   = []
            smoosh   = []
            for ele in batch:

                # Load image the first time then hold it in memory
                # this will slow down the first epoch, but will
                # speed up subsiquent epochs
                if isinstance(ele[IMAGE], str):
                    ele[IMAGE] = load_image((ele[IMAGE]))
                x.append(ele[IMAGE])
                steering.append(ele[STEERING])
                throttle.append(ele[THROTTLE])

                if mode in ["L", "C"]:
                    is_sim.append(ele[IS_SIM])
                    if mode == "C":
                        smoosh.append([0.5,0.5])
                    elif mode == "L":
                        smoosh.append(0.5)
                    else:
                        assert False, "POOP"
                    y = [steering, throttle, is_sim, smoosh]
                else:
                    y = [steering, throttle]

            yield [x], y


def get_gens(tub_paths, batch_size=32, train_frac=0.8, seed=42,
             mode  = "POOP"):
    np.random.seed(41)
    records = []
    for path in tub_paths:
        for r_path in Path(path).glob("record*.json"):
            with open(r_path, "r") as f:
                data = json.load(f)
                data[IMAGE] = str(Path(path) / data[IMAGE])
                records.append(data)
    np.random.shuffle(records)
    train_split = int(len(records) * train_frac)
    train_data = records[:train_split]
    valid_data = records[train_split:]

    train_gen = get_gen(train_data, batch_size, smooshing, categorical = categorical)
    valid_gen = get_gen(valid_data, batch_size, smooshing, categorical = categorical)

    return ((train_gen, int(len(train_data)/batch_size)),
           (valid_gen, int(len(valid_data)/batch_size)))


