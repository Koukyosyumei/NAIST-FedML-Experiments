import copy
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import wandb
from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_list_to_grad

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.distributed.fedavg.FedAVGAggregator import FedAVGAggregator
from fedml_api.distributed.fedavg.utils import transform_list_to_tensor


class FedAVGGradientAggregator(FedAVGAggregator):
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
        adversary_flag=None,
    ):
        super().__init__(
            train_global,
            test_global,
            all_train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            worker_num,
            device,
            args,
            model_trainer,
        )
        self.model_list_history = {c: [] for c in range(args.client_num_in_total)}
        self.round_idx = 0
        if adversary_flag is not None:
            self.adversary_idx = [i for i, f in enumerate(adversary_flag) if f == 1]
        else:
            self.adversary_idx = []

    def set_global_model_gradients(self, model_gradients, device, weight=1):
        self.trainer.set_model_gradients(model_gradients, device, weight=1)

    def anomalydetection(self, sender_id_to_client_index):
        pass

    def aggregate(self, sender_id_to_client_index):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_grad(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]
            self.model_list_history[sender_id_to_client_index[idx + 1]].append(
                (self.sample_num_dict[idx], self.model_dict[idx])
            )

        self.round_idx += 1
        if self.round_idx == self.args.comm_round:
            with open(
                f"{self.args.output_dir}/model_list_history.pickle", mode="wb"
            ) as f:
                logging.info("saving history")
                pickle.dump(self.model_list_history, f)

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        averaged_gradient = [
            torch.zeros(grad.shape).to(self.device) for grad in model_list[0][1]
        ]

        for i in range(0, len(model_list)):
            local_sample_number, local_gradient = model_list[i]
            w = local_sample_number / training_num
            for grad_idx in range(len(averaged_gradient)):
                averaged_gradient[grad_idx].data += (
                    local_gradient[grad_idx].data.to(self.device) * w
                )

        # update the global model which is cached at the server side
        self.set_global_model_gradients(averaged_gradient, self.device, weight=1)
        averaged_params = self.get_global_model_params()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if self.args.ignore_adversary == 0:
            candidates_idx = list(range(client_num_in_total))
        else:
            candidates_idx = list(
                set(list(range(client_num_in_total))) - set(self.adversary_idx)
            )

        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in candidates_idx]
        else:
            num_clients = min(client_num_per_round, len(candidates_idx))
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                candidates_idx, num_clients, replace=False
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.device,
            self.args,
        ):
            return

        if (
            round_idx % self.args.frequency_of_the_test == 0
            or round_idx == self.args.comm_round - 1
        ):
            logging.info(
                "################test_on_server_for_all_clients : {}".format(round_idx)
            )
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(
                    self.train_data_local_dict[client_idx], self.device, self.args
                )
                train_tot_correct, train_num_sample, train_loss = (
                    metrics["test_correct"],
                    metrics["test_total"],
                    metrics["test_loss"],
                )
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing.
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {"training_acc": train_acc, "training_loss": train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            logging.info(stats)

        if round_idx == self.args.comm_round - 1:
            logging.info("__finish aggregation")
            time.sleep(3)
            MPI.COMM_WORLD.Abort()
