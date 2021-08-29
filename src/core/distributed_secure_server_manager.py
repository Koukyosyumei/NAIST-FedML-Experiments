import logging
import os
import sys

from .utils import transform_grad_to_list

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))
from fedml_api.distributed.fedavg.FedAvgServerManager import FedAVGServerManager
from fedml_api.distributed.fedavg.message_define import MyMessage
from fedml_api.distributed.fedavg.utils import (
    post_complete_message_to_sweep_process,
    transform_tensor_to_list,
)
from fedml_core.distributed.communication.message import Message


class SecureFedAVGServerManager(FedAVGServerManager):
    def send_init_msg(self):
        self.sender_id_to_client_index = {}
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
        )
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id, global_model_params, client_indexes[process_id - 1]
            )
            self.sender_id_to_client_index[process_id] = client_indexes[process_id - 1]

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            sender_id - 1, model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.method == "RFFL":
                reward_gradients = self.aggregator.aggregate(
                    self.sender_id_to_client_index
                )
            else:
                global_model_params = self.aggregator.aggregate(
                    self.sender_id_to_client_index
                )

            logging.info("Start Anomaly Detection")
            self.aggregator.anomalydetection(self.sender_id_to_client_index)
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                print("here")
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.round_idx,
                    self.args.client_num_in_total,
                    self.args.client_num_per_round,
                )

            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)

            if self.args.method == "RFFL":
                for receiver_id in range(1, self.size):
                    params = reward_gradients[client_indexes[receiver_id - 1]]
                    if self.args.is_mobile == 1:
                        params = transform_grad_to_list(params)
                    self.send_message_sync_model_to_client(
                        receiver_id,
                        params,
                        client_indexes[receiver_id - 1],
                    )
            else:
                if self.args.is_mobile == 1:
                    global_model_params = transform_tensor_to_list(global_model_params)

                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(
                        receiver_id,
                        global_model_params,
                        client_indexes[receiver_id - 1],
                    )
