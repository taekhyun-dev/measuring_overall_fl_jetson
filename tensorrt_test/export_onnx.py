import torch
from torchvision.models import mobilenet_v3_small

# 1. Model Prepare
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = mobilenet_v3_small(weights=None).to(DEVICE)
model.eval()

# 2. Dummy Input Create
BATCH_SIZE = 32
dummy_input = torch.randn(BATCH_SIZE, 3, 32, 32).to(DEVICE)

# 3. ONNX Export
onnx_file_path = "mobilenet_v3_small.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,        
    opset_version=11,          
    do_constant_folding=True,
    input_names=['input'],  
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"ONNX Transfomation: {onnx_file_path}")