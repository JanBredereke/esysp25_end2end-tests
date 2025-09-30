import torch
from brevitas.onnx import export_finn_onnx
from torch.nn import Module


class ONNXExporter():
    def __init__(self, model: Module, load_path: str, save_path:str) -> None:
        self.model = model
        self.load_path = load_path
        self.save_path = save_path

    def export(self):
        self.model.load_state_dict(torch.load(self.load_path))
        export_finn_onnx(self.model, input_shape=(1, 3, 32, 32), export_path=self.save_path)