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
DEVICE = torch.device("cuda")

# --- TensorRT 로거 ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTInference:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.input_name = None
        self.output_name = None
        
        # 입출력 이름 찾기
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_name = name
            else:
                self.output_name = name

    def infer(self, torch_input_tensor):
        # [중요 1] 에러 해결: 동적 배치 크기 명시
        # 현재 들어온 텐서의 모양을 컨텍스트에 설정해줍니다.
        # (배치 사이즈가 32라면 (32, 3, 32, 32)로 설정됨)
        self.context.set_binding_shape(0, tuple(torch_input_tensor.shape))
        
        # [중요 2] 출력 텐서 준비 (GPU 상에 바로 생성)
        # TensorRT가 계산한 결과를 담을 빈 텐서를 GPU에 만듭니다.
        # 출력 크기 계산: (Batch, 1000) -> MobileNetV3 기본 출력
        # CIFAR10용으로 수정된 모델이라면 (Batch, 10)일 수 있으나, 
        # 여기서는 안전하게 엔진의 출력 바인딩 크기를 참조합니다.
        output_shape = tuple(self.context.get_binding_shape(1))
        output_tensor = torch.empty(output_shape, device=DEVICE, dtype=torch.float16) # FP16 엔진이므로 float16
        
        # [중요 3] Zero-Copy: 메모리 주소(Pointer)만 넘김
        # 데이터를 복사하지 않고, PyTorch 텐서의 메모리 주소를 TensorRT에게 알려줍니다.
        bindings = [int(torch_input_tensor.data_ptr()), int(output_tensor.data_ptr())]
        
        # 실행 (비동기)
        self.context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
        
        return output_tensor

# --- 데이터 준비 ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# drop_last=True: 배치가 32로 딱 떨어지게 설정
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

print(f"TensorRT 엔진 로드 중: {ENGINE_PATH}")
trt_model = TensorRTInference(ENGINE_PATH)

# --- 워밍업 ---
print("GPU 워밍업 중...")
# GPU 텐서를 바로 생성해서 워밍업
dummy_input = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE, dtype=torch.float16) # FP16 입력
for _ in range(50): # 충분히 워밍업
    _ = trt_model.infer(dummy_input)
torch.cuda.synchronize()
print("워밍업 완료.")

# --- 측정 시작 ---
print("측정 시작...")
total_time = 0
images_processed = 0

# 순수 연산 시간 측정을 위해 CUDA Event 사용
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

with torch.no_grad():
    for images, _ in test_loader:
        # 1. 입력을 바로 GPU로 보내고 FP16으로 변환 (여기까지는 데이터 로딩 비용)
        # memory_format=torch.channels_last는 TRT에서 자동으로 처리하므로 필수는 아니지만 도움될 수 있음
        images = images.to(DEVICE).half()
        
        # 2. 추론 (Zero-Copy)
        outputs = trt_model.infer(images)
        
        images_processed += BATCH_SIZE

end_event.record()
torch.cuda.synchronize()

total_gpu_time_ms = start_event.elapsed_time(end_event)
total_gpu_time_sec = total_gpu_time_ms / 1000.0

print(f"\n--- TensorRT(FP16 + Zero-Copy) 측정 결과 ---")
print(f"총 {images_processed}개 이미지 처리 완료.")
print(f"총 소요 시간: {total_gpu_time_sec:.4f} 초")
print(f"이미지당 평균 처리 시간: {(total_gpu_time_ms / images_processed):.4f} ms")
print(f"FPS: {images_processed / total_gpu_time_sec:.2f}")