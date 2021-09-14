import copy

constant_settings = {
    "node_type": "grid_short.q",
    "gpupernode": 1,
    "gpu_mapping_yaml": "gpu_mapping.yaml",
    "worker_num_pergpu": 5,
    "comm_round": 55,
    "epochs": 5,
    "clip_grad": 0,
    "max_norm": 1,
    "partition_method": "pow",
    "partition_alpha": 0.1,
    "frequency_of_the_test": 5,
    "ci": 0,
    "submit_script": 1,
    "py_file": "./distributed_main.py",
    "autoencoder_lr": 0.01,
    "autoencoder_epochs": 5,
    "warm_up": 5,
    "alpha": 0.95,
    "gamma": 0.5,
    "sparcity": 1,
    "remove": 1,
    "k": 0.005,
    "indicative_features": "all",
    "adversary_type": "inflator",
    "ignore_adversary": 0,
    "free_rider_strategy": "advanced-delta",
    "noise_amp": 0.001,
    "num_of_augmentation": 0,
    "multiple_accounts_split": 0.8,
}

variables_settings = {
    "inflator_lr_weight": 1.0,
    "autoencoder_type": "STD-DAGMM",
    "inv": 0,
}


def main():
    methods = [
        # "FedAvg",
        # "STD-DAGMM",
        "STD-NUM-DAGMM",
        "QI",
        "FoolsGold",
        # "INV-FoolsGold",
    ]
    # datasets = ["cifar10", "fed_shakespeare"]
    datasets = ["cifar10", "fed_shakespeare"]
    client_num = [50, 20]  # [50, 20]
    adversary_ratio = [0.2, 0.05]
    magnifications = [2, 10]
    # inflator_strategy = ["simple", "multiple_accounts", "data_augmentation"]
    inflator_strategy = ["simple"]
    small_batch = True

    for method_name in methods:

        if method_name == "FedAvg":
            variables_settings["method"] = "FedAvg"
        elif method_name == "STD-NUM-DAGMM":
            variables_settings["method"] = "AE"
            variables_settings["autoencoder_type"] = "STD-NUM-DAGMM"
        elif method_name == "STD-DAGMM":
            variables_settings["method"] = "AE"
            variables_settings["autoencoder_type"] = "STD-DAGMM"
        elif method_name == "FoolsGold":
            variables_settings["method"] = "FoolsGold"
            variables_settings["inv"] = 0
        elif method_name == "INV-FoolsGold":
            variables_settings["method"] = "FoolsGold"
            variables_settings["inv"] = 1
        elif method_name == "QI":
            variables_settings["method"] = "QI"

        for dataset in datasets:
            variables_settings["dataset"] = dataset
            for c_num in client_num:
                variables_settings["client_num"] = c_num
                if method_name == "QI":
                    variables_settings["client_num_per_round"] = int(c_num / 2)
                else:
                    variables_settings["client_num_per_round"] = int(c_num)
                for a_rat in adversary_ratio:

                    variables_settings["adversary_num"] = int(c_num * a_rat)

                    if dataset == "cifar10":
                        variables_settings["model"] = "resnet56"
                        variables_settings["lr"] = 0.001
                        variables_settings[
                            "data_dir"
                        ] = "/work/hideaki-t/dev/FedML/data/cifar10"
                        variables_settings["batch_size"] = 20
                        variables_settings["poor_adversary"] = 1
                        variables_settings["client_optimizer"] = "adam"
                    elif dataset == "fed_shakespeare":
                        variables_settings["model"] = "rnn"
                        variables_settings["lr"] = 1.47
                        variables_settings[
                            "data_dir"
                        ] = "/work/hideaki-t/dev/FedML/data/fed_shakespeare/datasets"
                        variables_settings["batch_size"] = 10
                        variables_settings["poor_adversary"] = 1
                        variables_settings["client_optimizer"] = "sgd"

                    for inf_strategy in inflator_strategy:
                        variables_settings["inflator_strategy"] = inf_strategy

                        if inf_strategy != "multiple_accounts":
                            temp_magnifications = copy.deepcopy(magnifications)
                        else:
                            temp_magnifications = [1]

                        for mag in temp_magnifications:
                            variables_settings["water_powered_magnification"] = mag
                            variables_settings["inflator_data_size"] = int(
                                (50000 / c_num) / mag
                            )

                            temp_mag = [1, mag] if small_batch else [1]
                            for i in list(set(temp_mag)):
                                variables_settings["inflator_batch_size"] = int(
                                    variables_settings["batch_size"] / i
                                )

                                if (
                                    variables_settings["inflator_strategy"]
                                    == "multiple_accounts"
                                    and variables_settings["adversary_num"] == 1
                                ):
                                    continue

                                config_name = (
                                    "config_autogenerated/"
                                    + method_name
                                    + "_".join(
                                        list(
                                            map(
                                                lambda x: str(x).replace("/", ""),
                                                list(variables_settings.values()),
                                            )
                                        )
                                    )
                                )
                                with open(config_name, mode="w") as f:
                                    for k, v in constant_settings.items():
                                        if type(v) == str:
                                            f.write(f'{k}="{v}"\n')
                                        else:
                                            f.write(f"{k}={v}\n")
                                    for k, v in variables_settings.items():
                                        if type(v) == str:
                                            f.write(f'{k}="{v}"\n')
                                        else:
                                            f.write(f"{k}={v}\n")


if __name__ == "__main__":
    main()
