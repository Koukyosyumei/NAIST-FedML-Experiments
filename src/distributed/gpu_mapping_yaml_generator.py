import argparse

import yaml


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="gpu_mapping.yaml",
        metavar="YP",
        help="the path to the yaml file",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=10,
        metavar="CNP",
        help="client_num_per_round",
    )
    parser.add_argument(
        "--worker_num_pernode",
        type=int,
        default=1,
        metavar="CNPN",
        help="the number of clients per node",
    )
    parser.add_argument(
        "--gpupernode",
        type=int,
        default=2,
        metavar="NPN",
        help="the number of GPUs per node",
    )

    return parser


def create_dict(client_num_per_round, worker_num_pernode, gpupernode, dict_key):
    worker_num = (client_num_per_round + 1) // (worker_num_pernode * gpupernode)
    worker_num = (
        worker_num + 1
        if (client_num_per_round + 1) % (worker_num_pernode * gpupernode) != 0
        else worker_num
    )

    mapping_dict = {}
    remaining = client_num_per_round + 1
    for worker_id in range(worker_num):
        if remaining >= worker_num_pernode * gpupernode:
            mapping_dict[f"gpu-worker{worker_id}"] = [worker_num_pernode] * gpupernode
            remaining -= worker_num_pernode * gpupernode
        else:
            temp_list = [0] * gpupernode
            num_used_node = remaining // worker_num_pernode
            for i in range(num_used_node):
                temp_list[i] = worker_num_pernode
            temp_list[num_used_node] = remaining % worker_num_pernode
            mapping_dict[f"gpu-worker{worker_id}"] = temp_list
            remaining -= (
                num_used_node * worker_num_pernode + remaining % worker_num_pernode
            )

    print(mapping_dict)
    assert remaining == 0

    return {dict_key: mapping_dict}


class MyDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description="create gpu-mapping yaml"))
    args = parser.parse_args()
    with open(args.yaml_path, "r") as yamlfile:
        cur_yaml = yaml.safe_load(yamlfile)
        dict_key = f"mapping_config_client_num_per_round_{args.client_num_per_round}_worker_num_pernode_{args.worker_num_pernode}_gpupernode_{args.gpupernode}"
        if dict_key not in cur_yaml:
            mapping_dict = create_dict(
                args.client_num_per_round,
                args.worker_num_pernode,
                args.gpupernode,
                dict_key,
            )
            sdump = "" + yaml.dump(
                mapping_dict, indent=4, Dumper=MyDumper, default_flow_style=False
            )
            print(sdump)
            with open(args.yaml_path, "a") as fo:
                fo.write(sdump)
