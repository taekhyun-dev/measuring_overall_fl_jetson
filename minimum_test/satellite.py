# minimum_test/satellite.py

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from datetime import datetime
from typing import Tuple, Dict
from ml.model import PyTorchModel, create_mobilenet
from ml.training import evaluate_model
from utils.logging_setup import KST
from config import LOCAL_EPOCHS, FEDPROX_MU

# ----- CLASS DEFINITION ----- #
class Satellite:
    def __init__ (self, sat_id: int, sim_logger, perf_logger, 
                  initial_model: PyTorchModel, train_loader, val_loader):
        self.sat_id = sat_id
        self.logger = sim_logger
        self.perf_logger = perf_logger
        self.local_model = initial_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.global_model = initial_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"SAT {self.sat_id} 생성")

    def _train_and_eval_measuring_time(self) -> Tuple[Dict, float, float]:
        temp_model = create_mobilenet()
        temp_model.load_state_dict(self.local_model.model_state_dict)
        temp_model.to(self.device)
        temp_model.train()

        global_model_ref = create_mobilenet()
        global_model_ref.load_state_dict(self.global_model.model_state_dict)
        global_model_ref.to(self.device)
        global_model_ref.eval()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
        for epoch in range(LOCAL_EPOCHS):
            self.logger.info(f"    - SAT {self.sat_id}: 에포크 {epoch+1}/{LOCAL_EPOCHS} 진행 중...")
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = temp_model(images)
                loss = criterion(outputs, labels)
                
                prox_term = 0.0
                for local_param, global_param in zip(temp_model.parameters(), global_model_ref.parameters()):
                    prox_term += torch.sum(torch.pow(local_param - global_param.detach(), 2))

                total_loss = loss + (FEDPROX_MU / 2) * prox_term

                total_loss.backward()
                optimizer.step()
            scheduler.step()
            
        new_state_dict = temp_model.cpu().state_dict()
        self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 완료 ({LOCAL_EPOCHS} 에포크). 검증 시작...")
            
        # --- 검증 파트 ---
        accuracy, loss = evaluate_model(new_state_dict, self.val_loader, self.device)
            
        return new_state_dict, accuracy, loss

    def train_and_eval_measuring_time(self):
        """CIFAR10 데이터셋으로 로컬 모델을 학습하고 검증"""
        self.state = 'TRAINING'
        self.logger.info(f"  ✅ SAT {self.sat_id}: 로컬 학습 시작 (v{self.local_model.version}).")
        new_state_dict = None
        try:
            new_state_dict, accuracy, loss = self._train_and_eval_measuring_time()
            self.local_model.model_state_dict = new_state_dict
            self.logger.info(f"  📊 [Local Validation] SAT: {self.sat_id}, Version: {self.local_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            self.perf_logger.info(f"{datetime.now(KST).isoformat()},LOCAL_VALIDATION,{self.sat_id},{self.local_model.version},N/A,{accuracy:.4f},{loss:.6f}")

            self.local_model.trained_by = [self.sat_id]
            self.model_ready_to_upload = True

        except Exception as e:
            self.logger.error(f"  💀 SAT {self.sat_id}: 학습 또는 검증 중 에러 발생 - {e}", exc_info=True)

        finally:
            self.logger.info(f"  🏁 SAT {self.sat_id}: 학습 절차 완료.")

    def receive_global_mode_measuring_timel(self, model: PyTorchModel):
        """지상국으로부터 글로벌 모델을 수신"""
        self.logger.info(f"  🛰️ SAT {self.sat_id}: 새로운 글로벌 모델 수신 (v{model.version}).")
        self.global_model = model
        self.local_model = model
        self.model_ready_to_upload = False

    def send_local_model_measuring_time(self) -> PyTorchModel | None:
        if self.model_ready_to_upload:
            self.model_ready_to_upload = False
            return self.local_model
        return None