import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_list_to_grad

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.distributed.fedavg.FedAvgClientManager import FedAVGClientManager
from fedml_api.distributed.fedavg.message_define import MyMessage
from fedml_api.distributed.fedavg.utils import post_complete_message_to_sweep_process


class RFFLClientManager(FedAVGClientManager):
    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_gradients = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            model_gradients = transform_list_to_grad(model_gradients)

        self.trainer.update_model_with_gradients(model_gradients)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num = self.trainer.train(self.round_idx)
        local_sample_num = int(local_sample_num * self.args.water_powered_magnification)
        self.send_model_to_server(0, weights, local_sample_num)
