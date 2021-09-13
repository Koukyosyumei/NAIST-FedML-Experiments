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
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


class FedAVGInflatorClientManager(ClientManager):
    def __init__(
        self,
        args,
        trainer,
        water_powered_magnification=1.0,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        adversary_idx=[],
    ):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

        self.adversary_idx = adversary_idx
        self.water_powered_magnification = water_powered_magnification
        self.client_index = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        logging.info(
            f"rank={self.rank}: handle_message_init_receive_model_from_server."
        )
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.client_index = int(client_index)

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
        logging.info(f"rank={self.rank}: handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.client_index = int(client_index)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.__train_with_inflation()
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train_with_inflation(self):
        logging.info(
            f"#######client_index={self.client_index} (rank={self.rank}): training with inflation########### round_id = {self.round_idx}"
        )
        weights, local_sample_num = self.trainer.train(self.round_idx)
        logging.info(f"adversary_idx = {self.adversary_idx}")
        if self.client_index in self.adversary_idx:
            logging.info(
                f"#######client_index={self.client_index} (rank={self.rank}) is an adversary: {self.water_powered_magnification}x"
            )
            local_sample_num = int(local_sample_num * self.water_powered_magnification)
        self.send_model_to_server(0, weights, local_sample_num)
