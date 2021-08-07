import argparse
import glob
import json
import pickle
import random

import numpy as np


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/work/hideaki-t/dev/FedML/data/MNIST",
        metavar="I",
        help="input_path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/work/hideaki-t/dev/NAIST-Experiments/data/label_flip",
        metavar="O",
        help="input_path",
    )

    parser.add_argument(
        "--flip_ratio",
        type=float,
        default=0.3,
        metavar="F",
        help="flip ratio",
    )

    return parser


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description="label flip"))
    args = parser.parse_args()

    # training-data
    with open(
        glob.glob(args.input_dir + "/train/*.json")[0],
        "r",
    ) as inf:
        cdata = json.load(inf)

    num_user = len(cdata["users"])
    print("num user: ", num_user)
    # print(cdata["users"])

    with open(args.output_dir + "/X_server.pickle", "wb") as inf:
        pickle.dump(cdata["user_data"]["f_00000"]["x"], inf)
    with open(args.output_dir + "/y_server.pickle", "wb") as inf:
        pickle.dump(cdata["user_data"]["f_00000"]["y"], inf)

    cdata["user_data"].pop("f_00000")
    cdata["users"] = cdata["users"][1:]
    cdata["num_samples"] = cdata["num_samples"][1:]
    cdata["quality"] = []

    for i, idx in enumerate(cdata["users"]):
        temp_array = np.array(cdata["user_data"][idx]["y"])
        data_size = len(temp_array)
        flip_size = int(random.uniform(0, 0.3) * data_size)
        temp_array[np.random.randint(0, data_size, (flip_size))] = np.random.randint(
            0, 10, (flip_size)
        )
        cdata["user_data"][idx]["y"] = temp_array.tolist()
        cdata["quality"].append(1 - flip_size / data_size)

    with open(
        args.output_dir + "/train/train_label_fliped.json",
        "w",
    ) as f:
        json.dump(cdata, f)
    with open(
        args.output_dir + "/credibility_train_label_fliped.pickle",
        "wb",
    ) as f:
        pickle.dump(cdata["quality"], f)

    # test
    with open(
        glob.glob(args.input_dir + "/test/*.json")[0],
        "r",
    ) as inf:
        cdata = json.load(inf)
    cdata["user_data"].pop("f_00000")
    cdata["users"] = cdata["users"][1:]
    cdata["num_samples"] = cdata["num_samples"][1:]
    cdata["quality"] = []
    with open(
        args.output_dir + "/test/test_label_fliped.json",
        "w",
    ) as f:
        json.dump(cdata, f)
