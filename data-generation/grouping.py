import argparse
import glob
import json


def agg(input_path, output_path, group_size):

    with open(
        input_path,
        "r",
    ) as inf:
        cdata = json.load(inf)

    users = cdata["users"]

    users_chunks = {
        f"fagg_{i}": users[s - group_size : s]
        for i, s in enumerate(
            list(range(group_size, len(users) + group_size, group_size))
        )
    }
    num_samples_chunks = [
        sum(cdata["num_samples"][s - group_size : s])
        for s in list(range(group_size, len(users) + group_size, group_size))
    ]
    user_data_chunks = {}

    for agg_id, ori_ids in users_chunks.items():
        user_data_chunks[agg_id] = {"y": [], "x": []}
        for ori_id in ori_ids:
            user_data_chunks[agg_id]["y"] += cdata["user_data"][ori_id]["y"]
            user_data_chunks[agg_id]["x"] += cdata["user_data"][ori_id]["x"]

    aggdata = {}
    aggdata["users"] = list(users_chunks.keys())
    aggdata["num_samples"] = num_samples_chunks
    aggdata["user_data"] = user_data_chunks

    with open(output_path, "w") as f:
        json.dump(aggdata, f)


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
        default="/work/hideaki-t/dev/NAIST-Experiments/data/grouped",
        metavar="O",
        help="input_path",
    )

    parser.add_argument(
        "--group_size",
        type=int,
        default=50,
        metavar="S",
        help="group size",
    )

    return parser


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description="grouping"))
    args = parser.parse_args()

    input_train_json_path = glob.glob(args.input_dir + "/train/*.json")[0]
    input_test_json_path = glob.glob(args.input_dir + "/test/*.json")[0]

    output_train_json_path = args.output_dir + f"/train/train_{args.group_size}.json"
    output_test_json_path = args.output_dir + f"/test/test_{args.group_size}.json"

    agg(input_train_json_path, output_train_json_path, args.group_size)
    agg(input_test_json_path, output_test_json_path, args.group_size)
