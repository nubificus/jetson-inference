import torch
import torch.nn as nn
import torchvision.models as models
import torch.onnx

# Wrap ResNet18 to include Softmax
class ResNet18WithSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet152(pretrained=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.base(x)
        probs = self.softmax(logits)
        return probs

# Instantiate model
model = ResNet18WithSoftmax()
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export with static shape and softmax
torch.onnx.export(
    model,
    dummy_input,
    "resnet18_softmax_static.onnx",
    input_names=['data'],
    output_names=['prob'],
    dynamic_axes=None,
    opset_version=11
)

print("✅ Exported 'resnet18_softmax_static.onnx' with softmax included.")
