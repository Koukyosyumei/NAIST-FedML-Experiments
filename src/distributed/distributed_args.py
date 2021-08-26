def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir", type=str, default="./../../../data/cifar10", help="data directory"
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="hetero",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=0.5,
        metavar="PA",
        help="partition alpha (default: 0.5)",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=1000,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--inflator_num",
        type=int,
        default=0,
        metavar="IN",
        help="number of inflators in a distributed cluster",
    )

    parser.add_argument(
        "--num_of_augmentation",
        type=int,
        default=0,
        metavar="NA",
        help="number of augmentation",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=4,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="adam", help="SGD with momentum; adam"
    )

    parser.add_argument(
        "--backend", type=str, default="MPI", help="Backend for Server and Client"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=0.0001
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=10,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--is_mobile",
        type=int,
        default=1,
        help="whether the program is running on the FedML-Mobile server side",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=5,
        help="the frequency of the algorithms",
    )

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument(
        "--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server"
    )

    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key",
        type=str,
        default="mapping_default",
        help="the key in gpu utilization file",
    )

    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")

    # custom arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="path to the dir for saving the outputs",
    )

    args = parser.parse_args()
    return args
