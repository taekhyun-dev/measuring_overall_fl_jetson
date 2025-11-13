import torch
import time
from torchvision.models import mobilenet_v3_small
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader

# --- 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
BATCH_SIZE = 32
# NUM_IMAGES 변수 제거 (전체 데이터 사용)

# --- 모델 준비 ---
model = mobilenet_v3_small(weights=None).to(DEVICE).half()
model.eval()  # 추론 모드로 설정

# --- 데이터 준비 (CIFAR10 테스트 세트: 총 10,000개) ---
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"측정 대상: CIFAR-10 테스트 세트 (총 {len(test_dataset)}개 이미지)")

# --- GPU 워밍업 ---
print("GPU 워밍업 중...")
dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
for _ in range(5):
    _ = model(dummy_input)

torch.cuda.synchronize() 
print("워밍업 완료. 실제 측정 시작.")

# --- 시간 측정 시작 ---
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

images_processed = 0
with torch.no_grad(): 
    for images, labels in test_loader:
        images = images.to(DEVICE).half()
        
        # 모델 연산
        outputs = model(images)
        
        images_processed += images.shape[0]

# --- 시간 측정 종료 ---
end_event.record()

# GPU 작업이 모두 끝날 때까지 동기화
torch.cuda.synchronize()

total_gpu_time_ms = start_event.elapsed_time(end_event)
total_gpu_time_sec = total_gpu_time_ms / 1000.0

print(f"\n--- 측정 결과 ---")
print(f"총 {images_processed}개 이미지 처리 완료.")
print(f"순수 GPU 연산 시간: {total_gpu_time_ms:.2f} ms ({total_gpu_time_sec:.4f} 초)")
print(f"이미지당 평균 처리 시간: {total_gpu_time_ms / images_processed:.4f} ms")