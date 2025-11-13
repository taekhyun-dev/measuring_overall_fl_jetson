import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader

# --- 설정 ---
ENGINE_PATH = "mobilenet_v3_small_fp16.engine"
BATCH_SIZE = 32

# --- TensorRT 로거 설정 ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# --- 엔진 로드 클래스 ---
class TensorRTInference:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # 입출력 버퍼 할당
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            # 배치 사이즈가 -1(동적)인 경우 설정
            if shape[0] == -1:
                shape = (BATCH_SIZE, *shape[1:])
            
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 호스트(CPU) 메모리
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 디바이스(GPU) 메모리
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})

    def infer(self, input_data):
        # 1. 입력 데이터를 호스트 버퍼에 복사
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # 2. 호스트 -> 디바이스 전송 (비동기)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. 추론 실행
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 4. 디바이스 -> 호스트 전송 (비동기)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # 5. 동기화 (완료 대기)
        self.stream.synchronize()
        
        return self.outputs[0]['host']

# --- 데이터 준비 ---
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
# drop_last=True: TensorRT는 배치 크기가 고정되어야 오류가 안 남
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

# --- 엔진 초기화 ---
print(f"TensorRT 엔진 로드 중: {ENGINE_PATH}")
trt_model = TensorRTInference(ENGINE_PATH)

# --- 워밍업 ---
print("GPU 워밍업 중...")
dummy_data = np.random.randn(BATCH_SIZE, 3, 32, 32).astype(np.float32)
for _ in range(10):
    trt_model.infer(dummy_data)
print("워밍업 완료.")

# --- 측정 시작 ---
print("측정 시작...")
total_time = 0
images_processed = 0

# PyCUDA 시간 측정이 아닌 Python time 사용 (전체 파이프라인 시간)
start_time = time.time()

for images, _ in test_loader:
    # PyTorch Tensor -> Numpy 변환
    input_numpy = images.numpy().astype(np.float32)
    
    # 추론
    output = trt_model.infer(input_numpy)
    
    images_processed += BATCH_SIZE

end_time = time.time()
total_time = end_time - start_time

print(f"\n--- TensorRT(FP16) 측정 결과 ---")
print(f"총 {images_processed}개 이미지 처리 완료.")
print(f"총 소요 시간: {total_time:.4f} 초")
print(f"이미지당 평균 처리 시간: {(total_time / images_processed) * 1000:.4f} ms")
print(f"FPS: {images_processed / total_time:.2f}")