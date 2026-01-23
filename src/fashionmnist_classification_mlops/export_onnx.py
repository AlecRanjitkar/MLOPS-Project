import torch

from fashionmnist_classification_mlops.model import FashionCNN

MODEL_PATH = "models/model.pth"
ONNX_PATH = "models/model.onnx"

model = FashionCNN()
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

print("✅ Exported model to ONNX")
