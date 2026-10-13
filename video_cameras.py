"""Recorded camera library with genuine YOLO inference and synchronized evidence.

Videos are analyzed locally once, then played at their native frame rate in the
browser. Detections are indexed by source timestamp, never by a demo timer.
"""

import csv
import hashlib
import io
import json
import math
import mimetypes
import os
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
ANALYSIS_VERSION = 11


def weapon_kind(label):
    label = re.sub(r"[^a-z]+", " ", str(label).lower()).strip()
    if any(word in label.split() for word in ("knife", "knives", "dagger")):
        return "knife"
    if any(word in label.split() for word in
           ("gun", "guns", "pistol", "handgun", "rifle", "firearm", "revolver", "shotgun")):
        return "gun"
    return None


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def discover_videos(directory=None):
    folders = [Path(directory)] if directory else [ROOT / n for n in ("examples", "samples")]
    return sorted({p.resolve() for folder in folders if folder.is_dir()
                   for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS},
                  key=lambda p: str(p).casefold())


def json_write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)


class PersonIDs:
    """Persistent public pseudonyms. Camera numbers never enter a person's ID."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()  # Recorded analysis and the live webcam both allocate IDs.
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def resolve(self, source_hash, track_id):
        key = f"{source_hash}:{track_id}"
        with self.lock:
            if key not in self.data:
                number = max((int(v.split("-")[1]) for v in self.data.values()), default=0) + 1
                self.data[key] = f"P-{number:04d}"
                json_write(self.path, self.data)
            return self.data[key]


# Validation thresholds. CCTV values are tuned for distant people in recorded
# footage. A webcam sees one person up close: the body is cut off by the frame,
# a held object looks large relative to it, and wrists often leave the frame.
# Inference sizes: CCTV magnifies tiny, distant weapons with 1280-pixel passes
# and overlapping tiles. Webcam subjects are large, so 640-pixel passes without
# tiles find the same objects several times faster.
CCTV_PROFILE = {"knife_conf": .8, "wrist_gap": .12, "compact": .3, "compact_moving": .45,
                "fallback_conf": None, "knife_observations": 3,
                "full_size": 1280, "tiles": True, "pose_size": 1280, "hand_size": 640}
WEBCAM_PROFILE = {"knife_conf": .7, "wrist_gap": .25, "compact": 1.0, "compact_moving": 1.0,
                  "fallback_conf": .5, "knife_observations": 2,
                  "full_size": 640, "tiles": False, "pose_size": 640, "hand_size": 416, "track_size": 640,
                  "max_people": 3}  # Crowded frames fall back to the CCTV thresholds.


class _Box:
    """The subset of an Ultralytics box that the weapon pipeline reads."""

    def __init__(self, xyxy, cls, conf):
        import torch
        self.xyxy, self.cls, self.conf = torch.tensor([xyxy]), torch.tensor(cls), torch.tensor(conf)


class KnifeAugmentedModel:
    """Serve guns from weapon.pt and knives from a dedicated knife model.

    Exposes weapon.pt's predict interface and class indices, so detect_weapons
    and associate_hands work unchanged.
    """

    def __init__(self, weapon, knife, knife_class):
        self.weapon, self.knife, self.knife_class = weapon, knife, knife_class
        self.names = weapon.names

    def predict(self, source, classes, conf, imgsz, verbose=False, device="cpu"):
        from types import SimpleNamespace

        batch = source if isinstance(source, list) else [source]
        others = [c for c in classes if c != self.knife_class]
        guns = (self.weapon.predict(batch, classes=others, conf=conf, imgsz=imgsz, verbose=verbose, device=device)
                if others else [None]*len(batch))
        knives = (self.knife.predict(batch, conf=conf, imgsz=min(imgsz, 640), verbose=verbose, device=device)
                  if self.knife_class in classes else [None]*len(batch))
        results = []
        for gun, knife in zip(guns, knives):
            boxes = [_Box(b.xyxy[0].tolist(), int(b.cls.item()), float(b.conf.item())) for b in (gun.boxes if gun else [])]
            boxes += [_Box(b.xyxy[0].tolist(), self.knife_class, float(b.conf.item())) for b in (knife.boxes if knife else [])]
            results.append(SimpleNamespace(boxes=boxes))
        return results


SIZE_KEYS = ("full_size", "tiles", "pose_size", "hand_size", "track_size")


def detect_weapons(model, image, people, classes, confidence, profile=CCTV_PROFILE, rejected=None):
    """Magnify small objects, then merge overlapping predictions in source pixels."""
    height, width = image.shape[:2]
    candidates = []
    regions = [(0, 0, width, height)]
    # Overlapping tiles preserve weapons that are tiny in a full CCTV frame.
    for row in range(2 if profile["tiles"] else 0):
        for col in range(2):
            regions.append((int(width*col*.4), int(height*row*.4),
                            min(width, int(width*(col*.4+.6))),
                            min(height, int(height*(row*.4+.6)))))
    # In sparse scenes, inspect each person and their outstretched hands.
    if len(people) <= 6:
        for person in people:
            x1, y1, x2, y2 = person["box"]
            w, h = x2-x1, y2-y1
            region = (max(0, int(x1-w*.7)), max(0, int(y1-h*.1)),
                      min(width, int(x2+w*.7)), min(height, int(y2+h*.1)))
            # Without tiles, a region covering most of the frame repeats the full pass.
            if profile["tiles"] or (region[2]-region[0])*(region[3]-region[1]) < .6*width*height:
                regions.append(region)
    tiles = 5 if profile["tiles"] else 1
    for index, (left, top, right, bottom) in enumerate(regions):
        crop = image[top:bottom, left:right]
        if not crop.size:
            continue
        size = profile["full_size"] if index == 0 else 960 if index < tiles else 640
        result = model.predict(crop, classes=list(classes), conf=confidence,
                               imgsz=size, verbose=False, device="cpu")[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            coords = [round(x1+left, 1), round(y1+top, 1), round(x2+left, 1), round(y2+top, 1)]
            # Reject the weapon model's occasional whole-person class confusion.
            if any(box_iou(coords, person["box"]) > .5 for person in people):
                if rejected is not None:
                    rejected.append((classes[int(box.cls.item())], float(box.conf.item()), "box covers a whole person"))
                continue
            candidates.append({"class": classes[int(box.cls.item())],
                               "conf": round(float(box.conf.item()), 4), "box": coords,
                               "id": None, "color": None, "weapon": True})
    merged = []
    for det in sorted(candidates, key=lambda d: -d["conf"]):
        if any(det["class"] == d["class"] and box_iou(det["box"], d["box"]) > .3 for d in merged):
            continue
        merged.append(det)
    plausible = []
    for det in merged:
        owner_id = weapon_owner(det, people)
        if det["class"] == "knife" and det["conf"] < max(profile["knife_conf"], confidence):
            if rejected is not None:
                rejected.append(("knife", det["conf"], f"below the {max(profile['knife_conf'], confidence):.0%} knife threshold"))
            continue
        det["person_id"] = owner_id
        plausible.append(det)
    return plausible


def associate_hands(pose_model, weapon_model, image, detections, people, classes, confidence, previous=(),
                    profile=CCTV_PROFILE, rejected=None):
    """Use visible wrists to distinguish the holder from a nearby bystander."""
    if not detections and not people:
        return []
    poses = pose_model.predict(image, imgsz=profile["pose_size"], conf=.2, verbose=False, device="cpu")[0]
    bodies = []
    for index, box in enumerate(poses.boxes.xyxy.tolist()):
        person = max(people, key=lambda p: box_iou(box, p["box"]), default=None)
        person_id = person["id"] if person and box_iou(box, person["box"]) > .2 else None
        wrists = [poses.keypoints.xy[index][k].tolist() for k in (9, 10)
                  if float(poses.keypoints.conf[index][k]) > .35]
        bodies.append({"box": box, "id": person_id, "wrists": wrists})
    # Inspect small hand regions even when the full-frame detector misses a
    # second weapon. This is especially useful for distant people in crowds.
    height, width = image.shape[:2]
    crops, offsets = [], []
    for body in bodies:
        body_height = body["box"][3]-body["box"][1]
        if body_height < 120:
            continue
        centers = []
        for x, y in body["wrists"]:
            if any(math.hypot(x-px, y-py) < .15*body_height for px, py in centers):
                continue
            centers.append((x, y))
            radius = max(48, int(body_height*.4))
            left, top = max(0, int(x-radius)), max(0, int(y-radius))
            right, bottom = min(width, int(x+radius)), min(height, int(y+radius))
            crop = image[top:bottom, left:right]
            if crop.size:
                crops.append(crop)
                offsets.append((left, top))
    candidates = list(detections)
    for start in range(0, len(crops), 4):
        results = weapon_model.predict(crops[start:start+4], classes=list(classes),
            imgsz=profile["hand_size"], conf=confidence, verbose=False, device="cpu")
        for result, (left, top) in zip(results, offsets[start:start+4]):
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                kind, score = classes[int(box.cls.item())], float(box.conf.item())
                if kind == "knife" and score < max(profile["knife_conf"], confidence):
                    if rejected is not None:
                        rejected.append(("knife", score, f"below the {max(profile['knife_conf'], confidence):.0%} knife threshold"))
                    continue
                candidates.append({"class": kind, "conf": round(score, 4),
                    "box": [round(x1+left, 1), round(y1+top, 1), round(x2+left, 1), round(y2+top, 1)],
                    "id": None, "color": None, "weapon": True})
    merged = []
    for det in sorted(candidates, key=lambda d: -d["conf"]):
        if not any(d["class"] == det["class"] and box_iou(d["box"], det["box"]) > .3 for d in merged):
            merged.append(det)
    verified = []
    for det in merged:
        x1, y1, x2, y2 = det["box"]
        best, distance = None, float("inf")
        for body in bodies:
            height = max(1, body["box"][3]-body["box"][1])
            for x, y in body["wrists"]:
                gap = math.hypot(max(x1-x, 0, x-x2), max(y1-y, 0, y-y2))/height
                if gap < distance:
                    best, distance = body, gap
        if best and distance < profile["wrist_gap"]:
            compact_limit = profile["compact"]
            for old in previous:
                if old["class"] == det["class"] and old.get("person_id") == best["id"]:
                    old_box = old["box"]
                    travel = math.hypot((x1+x2-old_box[0]-old_box[2])/2,
                                        (y1+y2-old_box[1]-old_box[3])/2)
                    if travel < .25*(best["box"][3]-best["box"][1]):
                        compact_limit = profile["compact_moving"]
            if max(x2-x1, y2-y1) > compact_limit*(best["box"][3]-best["box"][1]):
                if rejected is not None:
                    rejected.append((det["class"], det["conf"], "too large for the holder's body"))
                continue
            if best["id"] is None:
                if rejected is not None:
                    rejected.append((det["class"], det["conf"], "holder is not a tracked person"))
                continue
            # Keep the holder through adjacent, ambiguous hand overlaps.
            # Continuity applies only if that person's visible wrist remains nearby.
            for old in previous:
                if old["class"] != det["class"]:
                    continue
                prior = next((b for b in bodies if b["id"] == old.get("person_id")), None)
                if prior is None:
                    continue
                h = max(1, prior["box"][3]-prior["box"][1])
                old_box = old["box"]
                travel = math.hypot((x1+x2-old_box[0]-old_box[2])/2,
                                    (y1+y2-old_box[1]-old_box[3])/2)/h
                wrist_gap = min((math.hypot(max(x1-x, 0, x-x2), max(y1-y, 0, y-y2))/h
                                 for x, y in prior["wrists"]), default=1)
                if travel < .25 and wrist_gap <= min(.08, distance+.04):
                    best = prior
                    break
            det["person_id"] = best["id"]
            verified.append(det)
        elif profile["fallback_conf"] is not None and det["conf"] >= profile["fallback_conf"]:
            # Up close the wrist is often out of frame; use the nearest person.
            det["person_id"] = weapon_owner(det, people)
            verified.append(det)
        elif rejected is not None:
            rejected.append((det["class"], det["conf"], "no visible wrist next to it"))
    return verified


def stabilize_boxes(frames):
    """Remove isolated box jitter with a short centered temporal filter."""
    for index in range(1, len(frames)-1):
        before, frame, after = frames[index-1:index+2]
        if after["time"]-before["time"] > .3:
            continue
        old = {d["id"]: d for d in before["dets"] if d["class"] == "person" and d["id"]}
        new = {d["id"]: d for d in after["dets"] if d["class"] == "person" and d["id"]}
        for det in frame["dets"]:
            if det["class"] != "person" or det["id"] not in old or det["id"] not in new:
                continue
            a, b = old[det["id"]], new[det["id"]]
            det["box"] = [round(.2*x+.6*y+.2*z, 2) for x, y, z in zip(a["box"], det["box"], b["box"])]


def weapon_owner(weapon, people):
    """Associate visible evidence spatially; do not infer identity across cameras."""
    x = (weapon["box"][0]+weapon["box"][2])/2
    y = (weapon["box"][1]+weapon["box"][3])/2
    nearest, best = None, float("inf")
    for person in people:
        x1, y1, x2, y2 = person["box"]
        distance = math.hypot(max(x1-x, 0, x-x2), max(y1-y, 0, y-y2))
        normalized = distance/max(y2-y1, 1)
        if normalized < min(best, .5):
            nearest, best = person.get("id"), normalized
    return nearest


def clothing_color(image, box):
    import cv2
    import numpy as np

    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    # Center of the torso avoids most background and the subject's face.
    left, right = max(0, x1 + (x2-x1)//5), min(w, x2 - (x2-x1)//5)
    top, bottom = max(0, y1 + (y2-y1)//5), min(h, y1 + (y2-y1)//2)
    roi = image[top:bottom, left:right]
    if not roi.size:
        return "Unknown"
    hue, sat, value = np.median(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)
    if value < 55:
        return "Black"
    if sat < 35:
        return "White" if value > 180 else "Gray"
    if hue < 10 or hue > 170:
        return "Red"
    if hue < 35:
        return "Yellow"
    if hue < 85:
        return "Green"
    if hue < 135:
        return "Blue"
    return "Purple"


class FrameProcessor:
    """Per-frame tracking, person snapshots, and validated weapon events.

    Recorded analysis and the live webcam share this pipeline. Live frames
    arrive at the speed of inference, so ``adaptive`` widens the continuity
    windows to span a few samples; recorded analysis keeps fixed windows.
    """

    def __init__(self, camera, general, weapon, weapon_classes, pose, tracker, args,
                 person_ids, evidence, source_hash, fingerprint, adaptive=False, profile=CCTV_PROFILE):
        self.camera, self.general, self.weapon, self.pose = camera, general, weapon, pose
        self.weapon_classes, self.tracker, self.args = weapon_classes, tracker, args
        self.person_ids, self.evidence = person_ids, evidence
        self.source_hash, self.fingerprint, self.adaptive = source_hash, fingerprint, adaptive
        self.profile, self.active_profile, self.rejected = profile, profile, []
        self.selected_classes = [i for i, name in general.names.items()
                                 if name in ("person", "backpack", "handbag", "suitcase")]
        self.events, self.tracks = [], {}
        self.active_weapons, self.pending_weapons, self.previous_weapons = {}, {}, []
        self.last_time = None

    def process(self, image, source_time):
        import cv2

        hold, gap = .4, .6
        if self.adaptive and self.last_time is not None:
            interval = source_time - self.last_time
            hold, gap = max(hold, 1.5*interval), max(gap, 2.5*interval)
        self.last_time = source_time
        general, tracks = self.general, self.tracks
        result = general.track(image, persist=True, tracker=self.tracker,
                               classes=self.selected_classes, imgsz=self.profile.get("track_size", self.args.imgsz), conf=0.15,
                               verbose=False, device="cpu")[0]
        detections = []
        for box in result.boxes:
            label = general.names[int(box.cls.item())]
            coords = [round(v, 1) for v in box.xyxy[0].tolist()]
            confidence = round(float(box.conf.item()), 4)
            track_id = int(box.id.item()) if box.id is not None else None
            public_id = self.person_ids.resolve(self.source_hash, track_id) if label == "person" and track_id is not None else track_id
            color = clothing_color(image, coords) if label == "person" else None
            detection = {"class": label, "conf": confidence, "box": coords,
                         "id": public_id, "color": color, "weapon": False}
            detections.append(detection)
            if track_id is not None:
                track = tracks.setdefault(track_id, {"id": public_id, "class": label,
                    "first_seen": source_time, "last_seen": source_time,
                    "color": color, "frames": 0, "conf": confidence, "snapshot_score": 0})
                track.update(last_seen=source_time, frames=track["frames"]+1,
                             color=color or track["color"], conf=confidence)
                if label == "person":
                    left, top, right, bottom = map(int, coords)
                    h, w = image.shape[:2]
                    quality = confidence * math.sqrt(max(0, right-left)*max(0, bottom-top))
                    if left <= 2 or right >= w-2 or top <= 2 or bottom >= h-2:
                        quality *= .6
                    if quality > track["snapshot_score"]:
                        px, py = max(4, (right-left)//12), max(4, (bottom-top)//20)
                        crop = image[max(0, top-py):min(h, bottom+py), max(0, left-px):min(w, right+px)]
                        if crop.size:
                            cv2.imwrite(str(self.evidence / f"person-{public_id}.jpg"), crop,
                                        [cv2.IMWRITE_JPEG_QUALITY, 92])
                            track.update(snapshot_score=quality, snapshot_time=round(source_time, 4),
                                         snapshot_url=f"/api/people/{public_id}/snapshot.jpg")
        people = [d for d in detections if d["class"] == "person"]
        self.rejected = []
        profile = self.profile
        if len(people) > self.profile.get("max_people", len(people)):
            # Crowds use the CCTV thresholds at this camera's inference sizes.
            profile = {**CCTV_PROFILE, **{k: v for k, v in self.profile.items() if k in SIZE_KEYS}}
        self.active_profile = profile
        weapon_dets = detect_weapons(self.weapon, image, people, self.weapon_classes, self.args.weapon_conf,
                                     profile, self.rejected)
        weapon_dets = associate_hands(self.pose, self.weapon, image, weapon_dets, people,
                                     self.weapon_classes, self.args.weapon_conf,
                                     [d for t, d in self.previous_weapons if source_time-t <= hold],
                                     profile, self.rejected)
        self.previous_weapons = [(t,d) for t,d in self.previous_weapons if source_time-t <= hold]
        self.previous_weapons.extend((source_time,d) for d in weapon_dets)
        strongest = {}
        for det in weapon_dets:
            key = (det["class"], det.get("person_id"))
            detections.append(det)
            if key not in strongest or det["conf"] > strongest[key]["conf"]:
                strongest[key] = det
        for key, det in strongest.items():
            kind = det["class"]
            active = self.active_weapons.get(key)
            if active is None or source_time - active["last"] > 5.0:
                pending = self.pending_weapons.get(key)
                if pending is None or source_time - pending["last"] > gap:
                    pending = {"time": source_time, "last": source_time,
                               "det": dict(det), "image": image.copy(), "count": 1}
                    self.pending_weapons[key] = pending
                else:
                    pending["last"] = source_time
                    pending["count"] += 1
                # Require repeated observations of the same class and holder.
                # Moving hands can shift the small box between samples.
                if pending["count"] < (profile["knife_observations"] if kind == "knife" else 2):
                    continue
                first = pending["det"]
                eid = f"{self.fingerprint[:16]}-{kind}-{len(self.events)+1}"
                evidence_file = f"{eid}.jpg"
                annotated = pending["image"]
                x1, y1, x2, y2 = map(int, first["box"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 55, 245), 3)
                cv2.putText(annotated, f"{kind.upper()} {first['conf']:.0%}",
                            (max(5, x1), max(24, y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (80, 55, 245), 2)
                cv2.imwrite(str(self.evidence / evidence_file), annotated)
                event = {"id": eid, "camera": self.camera["id"], "filename": self.camera["filename"],
                         "class": kind, "type": f"{kind.title()} detected", "severity": "CRITICAL",
                         "time": round(pending["time"], 4), "end": round(source_time, 4),
                         "conf": first["conf"], "box": first["box"], "person_id": first.get("person_id"), "evidence_file": evidence_file,
                         "snapshot_url": f"/api/evidence/{eid}.jpg"}
                self.events.append(event)
                self.active_weapons[key] = {"last": source_time, "event": event}
                self.pending_weapons.pop(key, None)
            else:
                active["last"] = source_time
                active["event"]["end"] = round(source_time, 4)
        return detections


LIVE_CAMERA_ID = "CAM-LIVE"


class LiveCamera:
    """Analyze webcam frames sent by the browser with the recorded-camera pipeline."""

    def __init__(self, library):
        self.library = library
        self.lock = threading.Lock()
        self.models = None
        self.processor = None
        self.session_start = 0.0
        self.frames = []
        self.last_thumbnail = 0.0
        self.camera = {"id": LIVE_CAMERA_ID, "name": "Webcam", "filename": "webcam", "live": True,
                       "status": "queued", "error": None, "fps": 0, "duration": 0,
                       "width": 1280, "height": 720, "progress": 0, "samples": 0, "weapon_events": 0,
                       "video_url": None, "analysis_url": f"/api/cameras/{LIVE_CAMERA_ID}/analysis",
                       "thumbnail_url": f"/api/cameras/{LIVE_CAMERA_ID}/thumbnail.jpg"}

    @property
    def thumbnail(self):
        return self.library.cache / f"{LIVE_CAMERA_ID}-thumbnail.jpg"

    def load(self):
        """Load independent model instances; recorded analysis runs concurrently."""
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO

            if not self.thumbnail.exists():
                placeholder = np.full((360, 640, 3), 24, np.uint8)
                cv2.putText(placeholder, "LIVE WEBCAM", (190, 190), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (200, 200, 200), 2)
                cv2.imwrite(str(self.thumbnail), placeholder)
            args = self.library.args
            general, weapon, pose = YOLO(args.model), YOLO(args.weapon_model), YOLO(args.pose_model)
            wanted = set(getattr(args, "live_weapons", "knife").split(","))
            weapon_classes = {int(i): weapon_kind(label) for i, label in weapon.names.items()
                              if weapon_kind(label) in wanted}
            if not weapon_classes:
                raise ValueError(f"--live-weapons {','.join(sorted(wanted))} matches no weapon model class")
            knife_path = Path(getattr(args, "knife_model", "") or "")
            knife_class = next((i for i, kind in weapon_classes.items() if kind == "knife"), None)
            if knife_path.is_file() and knife_class is not None:
                weapon = KnifeAugmentedModel(weapon, YOLO(str(knife_path)), knife_class)
                print(f"{LIVE_CAMERA_ID} knives: {knife_path.name}", flush=True)
            tracker = args.tracker
            tracker_path = Path(tracker)
            if tracker_path.is_file():
                import yaml
                config = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))
                if config.get("tracker_type") == "botsort" and config.get("with_reid"):
                    config["proximity_thresh"] = .35
                    config["model"] = str(ROOT / "yolo26n-cls.pt")
                    tracker = str(self.library.cache / f"{LIVE_CAMERA_ID}-tracker.yaml")
                    Path(tracker).write_text(yaml.safe_dump(config), encoding="utf-8")
            self.models = (general, weapon, weapon_classes, pose, tracker)
            with self.library.lock:
                self.camera.update(status="ready", progress=100, error=None)
            print(f"{LIVE_CAMERA_ID} ready: webcam frames are analyzed live", flush=True)
        except Exception as exc:
            with self.library.lock:
                self.camera.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"{LIVE_CAMERA_ID} unavailable: {exc}", flush=True)

    def start_session(self):
        """Begin a new webcam session: fresh tracker state and person IDs."""
        with self.lock:
            self.processor = None
            self.frames = []
            self.session_start = time.monotonic()
        return self.status()

    def status(self):
        with self.library.lock:
            return {k: v for k, v in self.camera.items()}

    def analyze(self, payload):
        """Run the full pipeline on one JPEG frame. Returns None while busy."""
        import cv2
        import numpy as np

        if self.models is None:
            raise ValueError("Live detection models are not loaded yet")
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Frame is not a decodable image")
        if not self.lock.acquire(blocking=False):
            return None
        try:
            general, weapon, weapon_classes, pose, tracker = self.models
            if self.processor is None:
                session = f"live-{time.time_ns()}"
                general.predictor = None  # Fresh tracker state for each webcam session.
                self.session_start = time.monotonic()
                self.processor = FrameProcessor(
                    self.camera, general, weapon, weapon_classes, pose, tracker,
                    self.library.args, self.library.person_ids, self.library.evidence, session,
                    hashlib.sha256(session.encode()).hexdigest(), adaptive=True,
                    profile={**WEBCAM_PROFILE, "knife_conf": getattr(self.library.args, "live_knife_conf",
                                                                     WEBCAM_PROFILE["knife_conf"])})
            processor = self.processor
            source_time = time.monotonic() - self.session_start
            known = len(processor.events)
            detections = processor.process(image, source_time)
            # Live playback cannot retroactively hide unconfirmed candidates, so
            # only weapons belonging to a confirmed, ongoing event are shown.
            detections = [d for d in detections if not d["weapon"] or any(
                e["class"] == d["class"] and e.get("person_id") == d.get("person_id")
                and e["time"] <= source_time <= e["end"] + .0001 for e in processor.events)]
            frame = {"time": round(source_time, 4), "dets": detections}
            # Weapons seen this frame that still need repeated observations.
            pending = [{"class": kind, "conf": p["det"]["conf"], "count": p["count"],
                        "needed": processor.active_profile["knife_observations"] if kind == "knife" else 2}
                       for (kind, _), p in processor.pending_weapons.items() if p["last"] == source_time]
            rejected = [{"class": kind, "conf": round(conf, 4), "reason": reason}
                        for kind, conf, reason in processor.rejected]
            for item in sorted(rejected, key=lambda r: -r["conf"])[:3]:
                print(f"{LIVE_CAMERA_ID} {source_time:6.1f}s {item['class']} {item['conf']:.0%} "
                      f"ignored: {item['reason']}", flush=True)
            self.frames = (self.frames + [frame])[-3600:]
            height, width = image.shape[:2]
            if time.monotonic() - self.last_thumbnail > 5:
                cv2.imwrite(str(self.thumbnail), image)
                self.last_thumbnail = time.monotonic()
            tracks = [{k: v for k, v in t.items() if k != "snapshot_score"}
                      for t in processor.tracks.values()]
            with self.library.lock:
                self.camera.update(width=width, height=height, samples=self.camera["samples"]+1,
                                   weapon_events=self.camera["weapon_events"]+len(processor.events)-known)
                self.library.analyses[LIVE_CAMERA_ID] = {
                    "camera": LIVE_CAMERA_ID, "fingerprint": processor.fingerprint,
                    "sample_fps": self.library.args.analysis_fps, "live": True,
                    "frames": self.frames, "events": processor.events, "tracks": tracks}
            return {**frame, "camera": LIVE_CAMERA_ID, "width": width, "height": height,
                    "new_events": [dict(e) for e in processor.events[known:]], "tracks": tracks,
                    "pending": pending, "rejected": rejected}
        finally:
            self.lock.release()


class CameraLibrary:
    def __init__(self, args):
        import cv2

        self.args = args
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.runtime = Path(args.runtime_dir).resolve() if getattr(args, "runtime_dir", None) else ROOT / "runtime"
        self.cache = self.runtime / "analysis"
        self.evidence = self.runtime / "evidence"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.person_ids = PersonIDs(self.runtime / "person_ids.json")
        camera_path = self.runtime / "cameras.json"
        camera_ids = json.loads(camera_path.read_text(encoding="utf-8")) if camera_path.exists() else {}
        seed_path = ROOT / "camera_sources.json"
        seeds = json.loads(seed_path.read_text()) if seed_path.exists() else {}
        for relative, cid in seeds.items():
            camera_ids.setdefault(str((ROOT / relative).resolve()), cid)
        self.cameras = {}
        self.analyses = {}
        self.capabilities = []
        self.model_error = None
        self.incidents = []
        self.incident_path = self.runtime / "incidents.json"
        if self.incident_path.exists():
            try:
                self.incidents = json.loads(self.incident_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        for path in discover_videos(args.videos):
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            valid = cap.isOpened() and math.isfinite(fps) and fps > 0 and frames > 0
            key = str(path)
            if key not in camera_ids:
                number = max((int(cid.split("-")[1]) for cid in camera_ids.values()), default=0) + 1
                camera_ids[key] = f"CAM-{number:02d}"
            cid = camera_ids[key]
            camera = {"id": cid, "name": path.stem, "filename": path.name,
                      "path": path, "status": "queued" if valid else "error",
                      "error": None if valid else "Video cannot be decoded",
                      "fps": round(fps, 3) if valid else 0,
                      "duration": round(frames / fps, 3) if valid else 0,
                      "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      "progress": 0, "samples": 0, "weapon_events": 0,
                      "video_url": f"/api/cameras/{cid}/video",
                      "analysis_url": f"/api/cameras/{cid}/analysis",
                      "thumbnail_url": f"/api/cameras/{cid}/thumbnail.jpg"}
            ok, frame = cap.read()
            if ok:
                thumb = self.cache / f"{cid}-thumbnail.jpg"
                cv2.imwrite(str(thumb), frame)
            cap.release()
            self.cameras[cid] = camera
        json_write(camera_path, camera_ids)
        self.live = LiveCamera(self) if getattr(args, "webcam", True) else None
        if self.live:
            self.cameras[LIVE_CAMERA_ID] = self.live.camera
        # Keep real, persisted evidence only for the videos currently installed.
        self.incidents = [e for e in self.incidents
                          if e.get("camera") in self.cameras and
                          e.get("filename") == self.cameras[e["camera"]]["filename"]]

    def catalog(self):
        with self.lock:
            return {"cameras": [{k: v for k, v in c.items() if k != "path"}
                                for c in self.cameras.values()],
                    "live_camera": LIVE_CAMERA_ID if self.live else None,
                    "capabilities": self.capabilities, "model_error": self.model_error,
                    "sample_fps": self.args.analysis_fps,
                    "weapon_conf": self.args.weapon_conf,
                    "incidents": list(self.incidents),
                    "uptime": round(time.monotonic() - self.started)}

    def fingerprint(self, camera):
        files = [camera["path"], Path(self.args.model), Path(self.args.weapon_model),
                 Path(self.args.pose_model), ROOT / "yolo26n-cls.pt"]
        parts = [str(ANALYSIS_VERSION), camera["id"], str(self.args.analysis_fps),
                 str(self.args.weapon_conf), self.args.tracker, str(self.args.imgsz)]
        for path in files:
            stat = path.stat()
            parts.extend((str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns)))
        tracker_path = Path(self.args.tracker)
        if tracker_path.is_file():
            parts.append(tracker_path.read_text(encoding="utf-8"))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def start(self):
        threading.Thread(target=self.analyze_all, daemon=True, name="video-analysis").start()
        if self.live:
            threading.Thread(target=self.live.load, daemon=True, name="live-models").start()

    def analyze_all(self):
        try:
            import torch
            from ultralytics import YOLO

            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
            general = YOLO(self.args.model)
            weapon = YOLO(self.args.weapon_model)
            pose = YOLO(self.args.pose_model)
            weapon_classes = {int(i): weapon_kind(label) for i, label in weapon.names.items()
                              if weapon_kind(label)}
            self.capabilities = sorted(set(weapon_classes.values()))
            if not {"gun", "knife"}.issubset(self.capabilities):
                raise ValueError("Weapon model must include both gun/pistol and knife classes; "
                                 f"found {weapon.names}")
            print(f"Weapon classes: {weapon.names}; analyzing at {self.args.analysis_fps:g} FPS", flush=True)
            for camera in self.cameras.values():
                if self.stop.is_set():
                    break
                if camera["status"] == "error" or camera.get("live"):
                    continue
                try:
                    self.analyze_camera(camera, general, weapon, weapon_classes, pose)
                except Exception as exc:
                    with self.lock:
                        camera.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    print(f"{camera['id']} failed: {exc}", flush=True)
        except Exception as exc:
            with self.lock:
                self.model_error = f"{type(exc).__name__}: {exc}"
                for camera in self.cameras.values():
                    if camera["status"] != "error" and not camera.get("live"):
                        camera.update(status="error", error=self.model_error)
            print(f"Analysis unavailable: {self.model_error}", flush=True)

    def analyze_camera(self, camera, general, weapon, weapon_classes, pose):
        import cv2

        from tracking_cleanup import consolidate
        with camera["path"].open("rb") as source:
            source_hash = hashlib.file_digest(source, "sha256").hexdigest()
        fingerprint = self.fingerprint(camera)
        cache_file = self.cache / f"{fingerprint}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if all((self.evidence / e["evidence_file"]).exists() for e in data["events"]) and all(
                    (self.evidence / f"person-{t['id']}.jpg").exists()
                    for t in data["tracks"] if t["class"] == "person"):
                    data = consolidate(data, self.evidence, self.person_ids, source_hash, ROOT / "yolo26n-cls.pt")
                    json_write(cache_file, data)
                    with self.lock:
                        self.reconcile_incidents(data)
                        self.analyses[camera["id"]] = data
                        camera.update(status="ready", progress=100, samples=len(data["frames"]),
                                      weapon_events=len(data["events"]), analysis_revision=fingerprint)
                    print(f"{camera['id']} ready (cached): {camera['filename']}", flush=True)
                    return
            except (ValueError, KeyError, OSError):
                pass
        with self.lock:
            camera.update(status="analyzing", progress=0)
        general.reset_callbacks()
        general.predictor = None  # Isolate tracker state; public person IDs remain persistent.
        with camera["path"].open("rb") as source:
            source_hash = hashlib.file_digest(source, "sha256").hexdigest()
        cap = cv2.VideoCapture(str(camera["path"]))
        # Sparse scenes can rebind by appearance after a person moves out of
        # view. Crowds retain a proximity gate to avoid swapping similar suits.
        import yaml
        tracker = self.args.tracker
        tracker_path = Path(tracker)
        if tracker_path.is_file():
            config = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))
            if config.get("tracker_type") == "botsort" and config.get("with_reid"):
                ok, probe = cap.read()
                if ok:
                    count = len(general.predict(probe, classes=[0], imgsz=self.args.imgsz,
                                                conf=.25, verbose=False, device="cpu")[0].boxes)
                    config["proximity_thresh"] = .35
                config["model"] = str(ROOT / "yolo26n-cls.pt")
                tracker = str(self.cache / f"{camera['id']}-tracker.yaml")
                Path(tracker).write_text(yaml.safe_dump(config), encoding="utf-8")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sample_step = 1 / self.args.analysis_fps
        frames = []
        processor = FrameProcessor(camera, general, weapon, weapon_classes, pose, tracker,
                                   self.args, self.person_ids, self.evidence, source_hash, fingerprint)
        events, tracks = processor.events, processor.tracks
        frame_number, next_sample = 0, 0.0
        try:
            while not self.stop.is_set():
                ok = cap.grab()
                if not ok:
                    break
                source_time = frame_number / camera["fps"]
                frame_number += 1
                if source_time + 1e-6 < next_sample:
                    continue
                ok, image = cap.retrieve()
                if not ok:
                    break
                next_sample += sample_step
                detections = processor.process(image, source_time)
                frames.append({"time": round(source_time, 4), "dets": detections})
                with self.lock:
                    camera.update(progress=min(99, round(source_time / camera["duration"] * 100)),
                                  samples=len(frames), weapon_events=len(events))
        finally:
            cap.release()
        if self.stop.is_set():
            return
        # Discard isolated, unconfirmed low-score candidates from playback.
        for frame in frames:
            frame["dets"] = [d for d in frame["dets"] if not d["weapon"] or any(
                e["class"] == d["class"] and e.get("person_id") == d.get("person_id")
                and e["time"] <= frame["time"] <= e["end"] + .0001
                for e in events)]
        stabilize_boxes(frames)
        for track in tracks.values():
            track.pop("snapshot_score", None)
        data = {"camera": camera["id"], "fingerprint": fingerprint,
                "sample_fps": self.args.analysis_fps, "frames": frames,
                "events": events, "tracks": list(tracks.values())}
        data = consolidate(data, self.evidence, self.person_ids, source_hash, ROOT / "yolo26n-cls.pt")
        json_write(cache_file, data)
        with self.lock:
            self.reconcile_incidents(data)
            self.analyses[camera["id"]] = data
            camera.update(status="ready", progress=100, samples=len(frames), weapon_events=len(events), analysis_revision=fingerprint)
        print(f"{camera['id']} ready: {camera['filename']}; {len(frames)} samples; "
              f"{len(events)} weapon appearances", flush=True)

    def reconcile_incidents(self, data):
        """Retain review history while archiving unsupported old model alerts."""
        updated = []
        for incident in self.incidents:
            if incident["camera"] != data["camera"]:
                updated.append(incident)
                continue
            candidates = [e for e in data["events"] if e["class"] == incident["class"] and
                          e["time"] <= incident.get("end", incident["time"]) + .35 and
                          incident["time"] <= e["end"] + .35 and (
                              e["id"] == incident["id"] or (
                              e["time"]-.35 <= incident["time"] <= e["end"]+.35 and (
                              box_iou(e.get("box", [0]*4), incident.get("box", [0]*4)) > .05 or
                              any(abs(frame["time"]-incident["time"]) < .5 and any(
                                  d.get("weapon") and d["class"] == e["class"] and
                                  d.get("person_id") == e.get("person_id") and
                                  box_iou(d["box"], incident.get("box", [0]*4)) > .05
                                  for d in frame["dets"]) for frame in data.get("frames", [])))))]
            event = max(candidates, key=lambda e: box_iou(e.get("box", [0]*4),
                        incident.get("box", [0]*4)), default=None)
            if event:
                incident = {**incident, **event}
            elif incident.get("status") != "Confirmed":
                incident = {**incident, "status": "False Positive",
                            "review_note": "Unsupported by current validated analysis"}
            existing = next((e for e in updated if e["id"] == incident["id"]), None)
            if existing is None:
                updated.append(incident)
        self.incidents = updated
        json_write(self.incident_path, self.incidents)

    def activate(self, cid, event_id):
        with self.lock:
            event = next((e for e in self.analyses.get(cid, {}).get("events", [])
                          if e["id"] == event_id), None)
            if event is None:
                raise ValueError("Unknown detection event")
            existing = next((e for e in self.incidents if e["id"] == event_id), None)
            if existing:
                return existing
            incident = {**event, "status": "New", "detected_at": datetime.now().isoformat(timespec="seconds")}
            self.incidents.insert(0, incident)
            json_write(self.incident_path, self.incidents)
            return incident


def make_video_handler(library):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, ctype="application/json", headers=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def file(self, path, ctype=None):
            if not path.is_file():
                return self.reply(404, {"error": "File unavailable"})
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                if not match or not any(match.groups()):
                    return self.reply(416, b"", headers={"Content-Range": f"bytes */{size}"})
                a, b = match.groups()
                start = int(a) if a else max(0, size - int(b))
                end = min(size - 1, int(b)) if a and b else size - 1
                if start >= size or end < start:
                    return self.reply(416, b"", headers={"Content-Range": f"bytes */{size}"})
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with path.open("rb") as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = source.read(min(256 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            if path in ("/", "/raqeeb.html"):
                return self.file(ROOT / "raqeeb.html", "text/html; charset=utf-8")
            if path == "/app.js":
                return self.file(ROOT / "app.js", "text/javascript; charset=utf-8")
            if path in ("/ui.css", "/dashboard-ui.js"):
                return self.file(ROOT / path[1:], "text/css; charset=utf-8" if path.endswith(".css") else "text/javascript; charset=utf-8")
            if path in ("/api/cameras", "/api/state"):
                return self.reply(200, library.catalog())
            if path == "/api/incidents":
                return self.reply(200, library.catalog()["incidents"])
            if path.startswith("/api/evidence/"):
                name = path.rsplit("/", 1)[-1]
                if not re.fullmatch(r"[a-f0-9]{16}-(gun|knife)-\d+\.jpg", name):
                    return self.reply(404, {"error": "Evidence unavailable"})
                return self.file(library.evidence / name, "image/jpeg")
            person_match = re.fullmatch(r"/api/people/(P-\d{4,})/snapshot\.jpg", path)
            if person_match:
                return self.file(library.evidence / f"person-{person_match.group(1)}.jpg", "image/jpeg")
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "cameras"]:
                cid, action = parts[2:]
                camera = library.cameras.get(cid)
                if camera is None:
                    return self.reply(404, {"error": "Unknown camera"})
                if action == "video":
                    if camera.get("live"):
                        return self.reply(404, {"error": "The webcam streams from the browser"})
                    return self.file(camera["path"])
                if action == "thumbnail.jpg":
                    return self.file(library.cache / f"{cid}-thumbnail.jpg", "image/jpeg")
                if action == "analysis":
                    with library.lock:
                        analysis = library.analyses.get(cid)
                    return self.reply(200, analysis) if analysis else self.reply(202, {"status": camera["status"]})
                if action == "detections.csv":
                    with library.lock:
                        data = library.analyses.get(cid, {})
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(["camera", "source_seconds", "track_id", "class", "confidence", "x1", "y1", "x2", "y2", "color"])
                    for frame in data.get("frames", []):
                        for det in frame["dets"]:
                            writer.writerow([cid, frame["time"], det["id"], det["class"], det["conf"], *det["box"], det["color"]])
                    return self.reply(200, output.getvalue(), "text/csv", {"Content-Disposition": f'attachment; filename="{cid}-detections.csv"'})
            return self.reply(404, {"error": "Not found"})

        def do_OPTIONS(self):
            return self.reply(204, b"", headers={"Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"})

        def live_frame(self, length):
            live = getattr(library, "live", None)
            if live is None:
                return self.reply(404, {"error": "Webcam disabled"})
            if length <= 0 or length > 8 * 1024 * 1024:
                return self.reply(400, {"error": "Invalid frame size"})
            result = live.analyze(self.rfile.read(length))
            if result is None:
                return self.reply(409, {"error": "Previous webcam frame is still being analyzed"})
            return self.reply(200, result)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path == "/api/live/frame":
                    return self.live_frame(length)
                if self.path == "/api/live/start":
                    live = getattr(library, "live", None)
                    return self.reply(200, live.start_session()) if live else self.reply(404, {"error": "Webcam disabled"})
                if length <= 0 or length > 16384:
                    return self.reply(400, {"error": "Invalid request size"})
                data = json.loads(self.rfile.read(length))
                if self.path == "/api/incidents":
                    return self.reply(200, library.activate(data["camera"], data["event_id"]))
                if self.path == "/api/incidents/status":
                    if data["status"] not in ("New", "Under Review", "Investigating", "Confirmed", "False Positive", "Resolved"):
                        raise ValueError("Invalid status")
                    with library.lock:
                        incident = next((e for e in library.incidents if e["id"] == data["id"]), None)
                        if incident is None:
                            return self.reply(404, {"error": "Unknown incident"})
                        incident["status"] = data["status"]
                        json_write(library.incident_path, library.incidents)
                    return self.reply(200, incident)
                return self.reply(404, {"error": "Not found"})
            except (ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": str(exc)})
    return Handler


def serve_video_cameras(args):
    if not math.isfinite(args.analysis_fps) or not 1 <= args.analysis_fps <= 30:
        raise ValueError("--analysis-fps must be between 1 and 30")
    if not 0 < args.weapon_conf <= 1:
        raise ValueError("--weapon-conf must be between 0 and 1")
    library = CameraLibrary(args)
    server = ThreadingHTTPServer((args.host, args.port), make_video_handler(library))
    server.daemon_threads = True
    library.start()
    recorded = sum(not c.get("live") for c in library.cameras.values())
    print(f"Raqeeb: http://{args.host}:{args.port} — {recorded} recorded cameras"
          f"{' + live webcam (' + LIVE_CAMERA_ID + ')' if library.live else ''}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        library.stop.set()
        server.shutdown()
        server.server_close()
