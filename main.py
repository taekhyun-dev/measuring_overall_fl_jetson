from minimum_test.satellite import Satellite
from ml.data import get_cifar10_loaders
from ml.model import PyTorchModel, create_mobilenet
from utils.logging_setup import setup_loggers
from config import NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, NUM_WORKERS

if __name__ == "__main__":
    sim_logger, perf_logger = setup_loggers()

    client_loaders, val_loader, test_loader = get_cifar10_loaders(sim_logger=sim_logger, num_clients=NUM_CLIENTS, dirichlet_alpha=DIRICHLET_ALPHA,
                                                                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # 초기 글로벌 모델 생성
    initial_pytorch_model = create_mobilenet()
    initial_global_model = PyTorchModel(version=0, model_state_dict=initial_pytorch_model.state_dict())
    sat1 = Satellite(sat_id=1, sim_logger=None, perf_logger=None,
                    initial_model=initial_global_model,
                    train_loader=client_loaders[0], val_loader=val_loader)
    sat2 = Satellite(sat_id=2, sim_logger=None, perf_logger=None,
                initial_model=initial_global_model,
                train_loader=client_loaders[1], val_loader=val_loader)
    sat3 = Satellite(sat_id=3, sim_logger=None, perf_logger=None,
                initial_model=initial_global_model,
                train_loader=client_loaders[2], val_loader=val_loader)
    sat4 = Satellite(sat_id=4, sim_logger=None, perf_logger=None,
                initial_model=initial_global_model,
                train_loader=client_loaders[3], val_loader=val_loader)
    sat5 = Satellite(sat_id=5, sim_logger=None, perf_logger=None,
                initial_model=initial_global_model,
                train_loader=client_loaders[4], val_loader=val_loader)
    sat1.train_and_eval_measuring_time()
    sat2.train_and_eval_measuring_time()
    sat3.train_and_eval_measuring_time()
    sat4.train_and_eval_measuring_time()
    sat5.train_and_eval_measuring_time()