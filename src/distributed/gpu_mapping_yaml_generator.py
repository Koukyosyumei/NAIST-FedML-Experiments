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
        "--client_num",
        type=int,
        default=10,
        metavar="CN",
        help="the total number of clients",
    )
    parser.add_argument(
        "--client_num_pernode",
        type=int,
        default=1,
        metavar="CNPN",
        help="the number of clients per node",
    )
    parser.add_argument(
        "--npernode",
        type=int,
        default=2,
        metavar="NPN",
        help="the number of GPUs per node",
    )

    return parser


def create_dict(client_num, client_num_pernode, npernode, dict_key):
    worker_num = (client_num + 1) // (client_num_pernode * npernode)
    worker_num = (
        worker_num + 1
        if (client_num + 1) % (client_num_pernode * npernode) != 0
        else worker_num
    )

    mapping_dict = {}
    remaining = client_num + 1
    for worker_id in range(worker_num):
        if remaining >= client_num_pernode * npernode:
            mapping_dict[f"gpu-worker{worker_id}"] = [client_num_pernode] * npernode
            remaining -= client_num_pernode * npernode
        else:
            temp_list = [0] * npernode
            num_used_node = remaining // client_num_pernode
            for i in range(num_used_node):
                temp_list[i] = client_num_pernode
            temp_list[num_used_node] = remaining % client_num_pernode
            mapping_dict[f"gpu-worker{worker_id}"] = temp_list
            remaining -= (
                num_used_node * client_num_pernode + remaining % client_num_pernode
            )

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
        dict_key = f"mapping_config_client_num_{args.client_num}_client_num_pernode_{args.client_num_pernode}_npernode_{args.npernode}"
        if dict_key not in cur_yaml:
            mapping_dict = create_dict(
                args.client_num, args.client_num_pernode, args.npernode, dict_key
            )
            sdump = "" + yaml.dump(
                mapping_dict, indent=4, Dumper=MyDumper, default_flow_style=False
            )
            print(sdump)
            with open(args.yaml_path, "a") as fo:
                fo.write(sdump)
