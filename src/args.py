def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="resnet56",
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
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="adam", help="SGD with momentum; adam"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=0.001
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=10,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=10,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=10,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=5,
        help="the frequency of the algorithms",
    )

    parser.add_argument(
        "--free_rider_num",
        type=int,
        default=1,
        help="number of free riders",
    )

    parser.add_argument(
        "--free_rider_strategy",
        type=str,
        default="delta",
        help="number of free riders",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=10,
        help="alpha",
    )

    parser.add_argument(
        "--agg_weight",
        type=float,
        default=0.15,
        help="weight for aggregated gradient",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="gamma",
    )

    parser.add_argument("--method", type=str, default="QI", help="federated method")

    parser.add_argument("--freerider", action="store_true")
    parser.add_argument("--overstate", action="store_true")

    parser.add_argument("--remove", action="store_true")

    parser.add_argument("--use_sparsify", action="store_true")
    parser.add_argument("--use_reputation", action="store_true")

    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--ci", type=int, default=0, help="CI")

    return parser
