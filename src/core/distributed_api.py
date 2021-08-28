import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.distributed.fedavg.FedAVGAggregator import FedAVGAggregator
from fedml_api.distributed.fedavg.FedAvgClientManager import FedAVGClientManager
from fedml_api.distributed.fedavg.FedAvgServerManager import FedAVGServerManager
from fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer
from fedml_api.standalone.fedavg.my_model_trainer_classification import (
    MyModelTrainer as MyModelTrainerCLS,
)
from fedml_api.standalone.fedavg.my_model_trainer_nwp import (
    MyModelTrainer as MyModelTrainerNWP,
)
from fedml_api.standalone.fedavg.my_model_trainer_tag_prediction import (
    MyModelTrainer as MyModelTrainerTAG,
)


def FedML_Distributed_Custom_API(
    process_id,
    worker_number,
    device,
    comm,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    args,
    server_initializer,
    client_initializer,
    model_trainer,
    adversary_flag=None,
    preprocessed_sampling_lists=None,
):
    if process_id == 0:
        server_initializer.initialize(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            model_trainer,
            adversary_flag=adversary_flag,
            preprocessed_sampling_lists=preprocessed_sampling_lists,
        )
    else:
        client_initializer.initialize(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )


class Server_Initializer:
    def __init__(
        self,
        aggregator_class=FedAVGAggregator,
        server_manager_class=FedAVGServerManager,
    ):
        self.aggregator_class = aggregator_class
        self.server_manager_class = server_manager_class

    def initialize(
        self,
        args,
        device,
        comm,
        rank,
        size,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        model_trainer,
        adversary_flag=None,
        preprocessed_sampling_lists=None,
    ):
        model_trainer.set_id(-1)

        # aggregator
        worker_num = size - 1
        aggregator = self.aggregator_class(
            train_data_global,
            test_data_global,
            train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            worker_num,
            device,
            args,
            model_trainer,
            adversary_flag=adversary_flag,
        )

        # start the distributed training
        backend = args.backend
        if preprocessed_sampling_lists is None:
            server_manager = self.server_manager_class(
                args, aggregator, comm, rank, size, backend
            )
        else:
            server_manager = self.server_manager_class(
                args,
                aggregator,
                comm,
                rank,
                size,
                backend,
                is_preprocessed=True,
                preprocessed_client_lists=preprocessed_sampling_lists,
            )
        server_manager.send_init_msg()
        server_manager.run()


class Client_Initializer:
    def __init__(
        self, trainer_class=FedAVGTrainer, client_manager_class=FedAVGClientManager
    ):
        self.trainer_class = trainer_class
        self.client_manager_class = client_manager_class

    def initialize(
        self,
        args,
        device,
        comm,
        process_id,
        size,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):

        client_index = process_id - 1
        model_trainer.set_id(client_index)
        backend = args.backend
        trainer = self.trainer_class(
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            test_data_local_dict,
            train_data_num,
            device,
            args,
            model_trainer,
        )
        client_manager = self.client_manager_class(
            args, trainer, comm, process_id, size, backend
        )
        client_manager.run()
