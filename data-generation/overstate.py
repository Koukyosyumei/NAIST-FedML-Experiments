import argparse
import glob
import json
import pickle
import random


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
        "--inflated_client_num",
        type=int,
        default=1,
        metavar="ICN",
        help="the number of inflated clients",
    )

    parser.add_argument(
        "--inflated_rate",
        type=float,
        default=3.0,
        metavar="IR",
        help="inflated rate",
    )

    parser.add_argument(
        "--rate_bound",
        type=float,
        default=2.0,
        metavar="RB",
        help="bounr rate",
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

    front_user_id = "f_00000" if "f_00000" in cdata["user_data"] else "fagg_0"

    with open(args.output_dir + "/X_server.pickle", "wb") as inf:
        pickle.dump(cdata["user_data"][front_user_id]["x"], inf)
    with open(args.output_dir + "/y_server.pickle", "wb") as inf:
        pickle.dump(cdata["user_data"][front_user_id]["y"], inf)

    cdata["user_data"].pop(front_user_id)
    cdata["users"] = cdata["users"][1:]
    cdata["num_samples"] = cdata["num_samples"][1:]
    cdata["quality"] = []

    inflated_idx = random.sample(cdata["users"], args.inflated_client_num)

    for i, idx in enumerate(cdata["users"]):
        if idx in inflated_idx:
            data_size = len(cdata["user_data"][idx]["y"])
            cut_size = int(
                random.uniform(args.rate_bound, 1 / args.inflated_rate) * data_size
            )
            cdata["user_data"][idx]["y"] = cdata["user_data"][idx]["y"][:cut_size]
            cdata["user_data"][idx]["x"] = cdata["user_data"][idx]["x"][:cut_size]
            cdata["quality"].append(1 - cut_size / data_size)
        else:
            cdata["quality"].append(1)

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
    cdata["user_data"].pop(front_user_id)
    cdata["users"] = cdata["users"][1:]
    cdata["num_samples"] = cdata["num_samples"][1:]
    cdata["quality"] = []
    with open(
        args.output_dir + "/test/test_label_fliped.json",
        "w",
    ) as f:
        json.dump(cdata, f)
