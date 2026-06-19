
import mediapipe as mp
import cv2
import os
import csv
from mediapipe.tasks import python
import pyarrow as pa
import pyarrow.parquet as pq
from mediapipe.tasks.python import vision
#%%
PATH = "/Volumes/T7/elarna/elarna_video_sorted_2020_2021"
OUT_DIR = "/Volumes/T7/elarna/elarna_landmarks.parquet"
DB_PATH = "/Volumes/T7/elarna/elarna_landmarks.duckdb"

BATCH_SIZE = 5000 # num of files to flush

SKIP_N = 2 #hyperparameter to extract every n frame
#%%
LANDMARKS = [
    pa.field(f"{ax}{j}", pa.float32()) for j in range(21) for ax in ("x", "y")
]

SCHEMA = pa.schema([
    pa.field("filename", pa.string()),
    pa.field("frame", pa.int32()),
    pa.field("hand_label", pa.string()),
    *LANDMARKS,
])

#%%
def flush(buffer : list[dict], out_dir : str) -> None:
    if not buffer:
        return

    arrays = {field.name: [] for field in SCHEMA}
    for row in buffer:
        for name in arrays:
            arrays[name].append(row.get(name))
    table = pa.table(
        {name: pa.array(vals, type=SCHEMA.field(name).type ) for name, vals in arrays.items()}
    )
    pq.write_to_dataset(
        table,
        root_path = out_dir,
        partition_cols = ["filename"],
        existing_data_behavior="overwrite_or_ignore",
    )
    buffer.clear()
#%%
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(
    model_asset_path = model_path,
    delegate = python.BaseOptions.Delegate.CPU
)

options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_tracking_confidence=0.1,
)
#%%
os.makedirs(OUT_DIR, exist_ok=True)
buffer: list[dict] = []
#%%
with vision.HandLandmarker.create_from_options(options=options) as landmark:

    for root, dirs, files in os.walk(PATH):
        for file_name in sorted(files):
            if not file_name.endswith(".mp4"):
                continue
            if file_name.startswith("._"):
                continue
            full_path = os.path.join(root, file_name)
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"Could not open .mp4 {full_path}")
                continue

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % SKIP_N == 0:
                    if len(frame.shape) == 2:  # grayscale → convert
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.shape[2] == 4:  # BGRA → convert
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    mp_img = mp.Image(mp.ImageFormat.SRGB, data=frame)
                    res    = landmark.detect(mp_img)

                    if res.hand_landmarks:
                        for hand_idx, landmarks in enumerate(res.hand_landmarks):
                            wrist      = landmarks[0]  # ← fixed, always index 0
                            hand_label = res.handedness[hand_idx][0].display_name

                            row = {
                                "filename":  full_path,
                                "frame":  frame_idx,
                                "hand_label": hand_label,
                            }
                            for j, lm in enumerate(landmarks):
                                row[f"x{j}"] = lm.x - wrist.x
                                row[f"y{j}"] = lm.y - wrist.y

                            buffer.append(row)

                            if len(buffer) >= BATCH_SIZE:
                                flush(buffer, OUT_DIR)

                frame_idx += 1

            cap.release()
            flush(buffer, OUT_DIR)
            print(f"{file_name} done")

print("All videos processed")
#%%
import duckdb

con = duckdb.connect(DB_PATH)

# All frames for one video
df = con.execute("""
    SELECT * FROM landmarks
    WHERE filename = 'sign_001.mp4'
""").df()

# Aggregate across everything — runs in seconds on 36M rows
df = con.execute("""
    SELECT filename, hand_label, COUNT(*) as n_frames
    FROM landmarks
    GROUP BY filename, hand_label
    ORDER BY filename
""").df()
#%%
