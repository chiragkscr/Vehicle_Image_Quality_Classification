
import argparse
import torch
from model import MultiLabelModel


def export(args):
    device = torch.device("cpu")
    model = MultiLabelModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    print(f"[INFO] ONNX model exported → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model",  default="model_best.pth", help="Input .pth file")
    parser.add_argument("--output", default="model.onnx",     help="Output .onnx file")
    export(parser.parse_args())