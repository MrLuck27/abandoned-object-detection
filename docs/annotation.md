# Annotation Specifications & Formats

This document defines the two annotation paradigms used by this project:
1. **Frame-Level Bounding Box Format (YOLO)** — For spatial object detector training.
2. **Video Event-Level Annotation Format (JSON)** — For temporal reasoning and alert verification.

---

## 1. Frame-Level Detection Annotations (YOLO Format)

Each image `frame_000123.jpg` is accompanied by a text file `frame_000123.txt` located in the corresponding `labels/` directory.

### Structure:
```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: Integer class identifier
  - `0`: Person
  - `1`: Bag (backpack, suitcase, handbag, luggage)
- `x_center`, `y_center`: Centroid coordinates normalized by image width and height [0.0, 1.0].
- `width`, `height`: Box dimensions normalized by image width and height [0.0, 1.0].

### Example (`frame_000123.txt`):
```
0 0.452300 0.582100 0.124500 0.385200
1 0.491200 0.724100 0.082000 0.114300
```

---

## 2. Event-Level Ground Truth Format (JSON)

Event-level annotations describe the temporal lifecycle of luggage items and human-object interactions throughout a surveillance sequence.

### Specification:
```json
{
  "video_filename.mp4": {
    "video": "video_filename.mp4",
    "duration_seconds": 30.0,
    "events": [
      {
        "object_id": "B01",
        "owner_id": "P01",
        "separation_start": 6.5,
        "abandoned_at": 21.5,
        "recovered_at": null,
        "event_type": "ABANDONED"
      }
    ]
  }
}
```

### Fields:
- `object_id`: Ground truth identifier of the bag/luggage.
- `owner_id`: Ground truth identifier of the person associated with the bag.
- `separation_start`: Timestamp (seconds) when the owner initiates departure beyond separation distance.
- `abandoned_at`: Timestamp (seconds) when the bag officially qualifies as abandoned (i.e. separation duration exceeds the configured abandonment time threshold, e.g. 15s).
- `recovered_at`: Timestamp (seconds) when the owner returns or bag is picked up (or `null` if not recovered).
- `event_type`: Categorical label:
  - `ABANDONED`: Bag remained unattended for required duration.
  - `NOT_ABANDONED`: Bag remained attended throughout.
  - `RECOVERED`: Owner departed but returned before threshold.
  - `FALSE_ALARM`: Unattended trigger on an attended bag.
