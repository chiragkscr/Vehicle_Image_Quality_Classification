import argparse
import os
import json
import time

import torch
import torchvision.transforms as T
from PIL import Image

from model import MultiLabelModel


CLASSES = ["blur", "low_light"]



def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])



def load_torch_model(path, device):
    model = MultiLabelModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_onnx_model(path):
    import onnxruntime as ort
    return ort.InferenceSession(path)


def predict_torch(model, img, device):
    with torch.no_grad():
        out = model(img.to(device))
        probs = torch.sigmoid(out)[0]
    return probs.cpu().numpy()


def predict_onnx(session, img):
    import numpy as np
    out = session.run(None, {"input": img.numpy()})[0]
    probs = 1 / (1 + np.exp(-out))
    return probs[0]



def save_json(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")


    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--out_json", default="output.json")

    args = parser.parse_args()

    device = torch.device("cpu")
    transform = get_transform()

    if args.backend == "torch":
        model = load_torch_model(args.model, device)
        infer = lambda img: predict_torch(model, img, device)
    else:
        session = load_onnx_model(args.model)
        infer = lambda img: predict_onnx(session, img)


    if os.path.isdir(args.input):
        paths = [
            os.path.join(args.input, x)
            for x in os.listdir(args.input)
            if x.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
    else:
        paths = [args.input]

    results = []
    total_infer_time = 0.0

    for path in paths:
        image = Image.open(path).convert("RGB")
        img = transform(image).unsqueeze(0)


        _ = infer(img)


        start = time.perf_counter()
        probs = infer(img)
        end = time.perf_counter()

        infer_time = end - start
        total_infer_time += infer_time


        max_idx = int(probs.argmax())
        confidence = float(probs[max_idx])
        predicted_class = CLASSES[max_idx]

        status = "Reject" if confidence > args.threshold else "Accept"

        results.append({
            "image": os.path.basename(path),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "status": status,
            "inference_time_sec": infer_time
        })

    avg_time = total_infer_time / len(results)

    save_json(results, args.out_json)

    print(f"[INFO] Processed {len(results)} images")
    print(f"[INFO] JSON saved → {args.out_json}")
    print(f"[INFO] Avg inference time: {avg_time:.6f} sec")
    print(f"[INFO] Total inference time: {total_infer_time:.4f} sec")


if __name__ == "__main__":
    main()