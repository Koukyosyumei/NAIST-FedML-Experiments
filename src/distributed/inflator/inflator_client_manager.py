import copy
import logging
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.distributed.fedavg.FedAvgClientManager import FedAVGClientManager
from fedml_api.distributed.fedavg.message_define import MyMessage
from fedml_api.distributed.fedavg.utils import (
    post_complete_message_to_sweep_process,
    transform_list_to_tensor,
)
from fedml_core.distributed.communication.message import Message


class FedAVGInflatorClientManager(FedAVGClientManager):
    def __init__(
        self,
        args,
        trainer,
        water_powered_magnification=1.0,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
    ):
        super().__init__(
            args, trainer, comm=comm, rank=rank, size=size, backend=backend
        )

        self.water_powered_magnification = water_powered_magnification
        logging.info(
            f"#######initialization########### inflation = {self.water_powered_magnification}"
        )

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.__train_with_inflation()

    def start_training(self):
        self.round_idx = 0
        self.__train_with_inflation()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.__train_with_inflation()
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def __train_with_inflation(self):
        logging.info(
            "#######training with inflation########### round_id = %d" % self.round_idx
        )
        weights, local_sample_num = self.trainer.train(self.round_idx)
        local_sample_num = int(local_sample_num * self.water_powered_magnification)
        self.send_model_to_server(0, weights, local_sample_num)
