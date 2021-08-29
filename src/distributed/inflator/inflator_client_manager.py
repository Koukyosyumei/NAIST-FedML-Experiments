import copy
import logging
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.distributed.fedavg.FedAvgClientManager import FedAVGClientManager


class FedAVGInflatorClientManager(FedAVGClientManager):
    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num = self.trainer.train(self.round_idx)
        local_sample_num = int(local_sample_num * self.args.water_powered_magnification)
        self.send_model_to_server(0, weights, local_sample_num)
