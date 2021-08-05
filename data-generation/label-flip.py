import copy
import glob
import json
import os
import pickle
import random

import numpy as np
import torch

if __name__ == "__main__":

    # training-data
    with open(
        "/work/hideaki-t/dev/FedML/data/MNIST/train/all_data_0_niid_0_keep_10_train_9.json",
        "r",
    ) as inf:
        cdata = json.load(inf)

    num_user = len(cdata["users"])
    print("num user: ", num_user)
    # print(cdata["users"])

    with open(
        "/work/hideaki-t/dev/NAIST-Experiments/data/label_flip/X_server.pickle", "wb"
    ) as inf:
        pickle.dump(cdata["user_data"]["f_00000"]["x"], inf)
    with open(
        "/work/hideaki-t/dev/NAIST-Experiments/data/label_flip/y_server.pickle", "wb"
    ) as inf:
        pickle.dump(cdata["user_data"]["f_00000"]["y"], inf)

    cdata["user_data"].pop("f_00000")
    cdata["users"] = cdata["users"][1:]
    cdata["num_samples"] = cdata["num_samples"][1:]
    cdata["quality"] = []

    for i, idx in enumerate(cdata["users"]):
        temp_array = np.array(cdata["user_data"][idx]["y"])
        data_size = len(temp_array)
        flip_size = int(data_size * (i + 1) / num_user)
        temp_array[np.random.randint(0, data_size, (flip_size))] = np.random.randint(
            0, 10, (flip_size)
        )
        cdata["user_data"][idx]["y"] = temp_array.tolist()
        cdata["quality"].append(1 - flip_size / data_size)

    with open(
        "/work/hideaki-t/dev/NAIST-Experiments/data/label_flip/train/train_label_fliped.json",
        "w",
    ) as f:
        json.dump(cdata, f)
    with open(
        "/work/hideaki-t/dev/NAIST-Experiments/data/label_flip/credibility_train_label_fliped.pickle",
        "wb",
    ) as f:
        pickle.dump(cdata["quality"], f)

    # test
    with open(
        "/work/hideaki-t/dev/FedML/data/MNIST/test/all_data_0_niid_0_keep_10_test_9.json",
        "r",
    ) as inf:
        cdata = json.load(inf)
    cdata["user_data"].pop("f_00000")
    cdata["users"] = cdata["users"][1:]
    cdata["num_samples"] = cdata["num_samples"][1:]
    cdata["quality"] = []
    with open(
        "/work/hideaki-t/dev/NAIST-Experiments/data/label_flip/test/test_label_fliped.json",
        "w",
    ) as f:
        json.dump(cdata, f)
