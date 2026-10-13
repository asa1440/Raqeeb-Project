"""Train knife.pt, a dedicated knife detector for the live webcam, on a GPU.

Downloads the photos listed in knife_dataset.json (Open Images knives plus
knife-free phones, pens, scissors, remotes, tools, and hands), fine-tunes
yolo26n.pt, compares the result with weapon.pt on held-out photos, and saves
knife.pt next to this script.

    .venv\\Scripts\\python.exe train_knife.py
"""

import argparse
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def download(manifest, out):
    import cv2
    import numpy as np

    def fetch(job):
        split, (source, image_id, boxes) = job
        image_path = out / "images" / split / f"{image_id}.jpg"
        label_path = out / "labels" / split / f"{image_id}.txt"
        if image_path.exists() and label_path.exists():
            return True
        url = manifest["image_url"].format(split=source, id=image_id)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                break
            except OSError:
                time.sleep(2 ** attempt)
        else:
            return False
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return False
        height, width = image.shape[:2]
        scale = min(1, 800 / max(height, width))
        if scale < 1:
            image = cv2.resize(image, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        label_path.write_text("".join(f"0 {cx} {cy} {w} {h}\n" for cx, cy, w, h in boxes))
        return True

    jobs = []
    for split, items in manifest["splits"].items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        jobs += [(split, item) for item in items]
    done = failed = 0
    with ThreadPoolExecutor(16) as pool:
        for ok in pool.map(fetch, jobs):
            done, failed = done + ok, failed + (not ok)
            if (done + failed) % 250 == 0:
                print(f"  downloaded {done + failed}/{len(jobs)}", flush=True)
    print(f"Dataset ready: {done} photos ({failed} unavailable) in {out}", flush=True)
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\n"
                                   "val: images/val\nnames:\n  0: knife\n", encoding="utf-8")
    return out / "data.yaml"


def evaluate(model_path, dataset, device):
    """Share of knife photos found and of knife-free photos falsely flagged."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    knife_ids = [i for i, name in model.names.items() if "knife" in name.lower()]

    def iou(a, b):
        inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

    rows = []
    for image in sorted((dataset / "images" / "val").glob("*.jpg")):
        truth = []
        for line in (dataset / "labels" / "val" / f"{image.stem}.txt").read_text().splitlines():
            _, cx, cy, w, h = map(float, line.split())
            truth.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
        result = model.predict(str(image), classes=knife_ids, conf=.1, imgsz=640, verbose=False, device=device)[0]
        rows.append((truth, list(zip(result.boxes.conf.tolist(), result.boxes.xyxyn.tolist()))))
    knives = [r for r in rows if r[0]]
    clear = [r for r in rows if not r[0]]
    table = {}
    for threshold in (.4, .5, .6, .7, .8):
        found = sum(any(c >= threshold and iou(t, b) >= .3 for c, b in dets for t in truth) for truth, dets in knives)
        false = sum(any(c >= threshold for c, _ in dets) for _, dets in clear)
        table[threshold] = (found / len(knives), false / len(clear))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "dataset" / "knife"), help="where to store the photos")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="lower to 8 if the GPU runs out of memory")
    parser.add_argument("--device", default="0", help="GPU index, or cpu")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see a GPU. Install the CUDA build of PyTorch (see README), "
                         "or pass --device cpu to train slowly on the processor.")
    if args.device != "cpu":
        print(f"Training on {torch.cuda.get_device_name(int(args.device))}", flush=True)
    dataset = Path(args.dataset)
    data = download(json.loads((ROOT / "knife_dataset.json").read_text(encoding="utf-8")), dataset)
    YOLO(str(ROOT / "yolo26n.pt")).train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        patience=15, close_mosaic=10, project=str(ROOT / "runs"), name="knife", exist_ok=True)
    best = ROOT / "runs" / "knife" / "weights" / "best.pt"
    if not best.exists():
        best = next((ROOT / "runs").rglob("knife/weights/best.pt"))

    print("\nHeld-out comparison (knife photos found / knife-free photos falsely flagged):", flush=True)
    new, old = evaluate(best, dataset, args.device), evaluate(ROOT / "weapon.pt", dataset, args.device)
    print("threshold   new knife model      weapon.pt")
    for threshold in new:
        (nf, nx), (of, ox) = new[threshold], old[threshold]
        print(f"   {threshold:.0%}      {nf:6.1%} / {nx:5.1%}      {of:6.1%} / {ox:5.1%}")
    better = sum(new[t][0] > old[t][0] and new[t][1] <= old[t][1] + .01 for t in new)
    if better >= 3:
        shutil.copy(best, ROOT / "knife.pt")
        print(f"\nSaved {ROOT / 'knife.pt'}. Restart server.py; the webcam now detects knives with it.")
    else:
        print(f"\nThe new model is not clearly better, so knife.pt was not replaced. It is at {best}.")


if __name__ == "__main__":
    main()
