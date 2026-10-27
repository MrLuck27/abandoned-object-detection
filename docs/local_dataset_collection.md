# Local Campus Dataset Collection Protocol & Annotation Guide

This document defines the standard operating procedure for recording, organizing, and annotating staged campus surveillance video sequences for unattended object detection.

---

## 1. Ethical, Privacy, and Consent Directives

1. **Informed Consent:** All actors appearing in staged videos must provide prior informed consent.
2. **No Facial Recognition:** No face recognition, biometric extraction, or identity tracking algorithms are permitted. Faces should be blurred or filmed from surveillance angles (mounted high at 2.5m - 4.0m) where facial features are not resolvable.
3. **Public Non-Interference:** Record in campus lobbies, seminar halls, or corridors during scheduled off-hours to prevent disrupting students or faculty.

---

## 2. Experimental Scenario Matrix

Collect videos (20–60 seconds each) across these 10 distinct categories:

| Scenario ID | Category | Description | Expected System Output |
| :---: | :--- | :--- | :--- |
| **SC-01** | `TRUE_ABANDONED` | Owner enters with bag, sets it on floor/bench, walks away and exits frame. Bag remains stationary > 15s. | 🚨 **Alert** (`ABANDONED`) |
| **SC-02** | `OWNER_RETURNS` | Owner sets bag down, walks away (separation begins), but returns within 10s (before threshold). | **No Alert** (`RECOVERED` -> `ATTENDED`) |
| **SC-03** | `NORMAL_ATTENDED` | Owner stands or sits beside their bag throughout the entire sequence. | **No Alert** (`ATTENDED`) |
| **SC-04** | `PASSERBY_CROWD` | Bag is left stationary; multiple passersby walk directly past or around the bag without claiming it. | **Verify Ownership**: Passersby should NOT be wrongly assigned as owner. |
| **SC-05** | `TEMPORARY_OCCLUSION` | Another person or cart passes directly in front of unattended bag, occluding it for 1–2 seconds. | **Grace Period**: State and timer must NOT reset prematurely. |
| **SC-06** | `BAG_RELOCATED` | Bag is picked up or moved by owner or someone else before abandonment threshold. | **Reset Timer** (`RECOVERED`) |
| **SC-07** | `MULTIPLE_BAGS` | Two owners place different bags in the scene; one owner leaves while the other remains. | Independent tracking: only the separated bag triggers alert. |
| **SC-08** | `LOW_ILLUMINATION` | Corridor with dim evening lighting or back-lit corridor glare. | Test detector robustness under low contrast. |
| **SC-09** | `LONG_RANGE_PERSPECTIVE` | Bag placed far from camera (small pixel area < 20x20 px). | Test small object detection limit. |
| **SC-10** | `MULTI_PERSON_CHAT` | Group of 3 people talking around luggage, then all 3 depart together. | Group ownership association -> departure -> alert. |

---

## 3. Camera Setup & Recording Specifications

- **Mounting Height:** 2.5 meters to 3.5 meters (standard surveillance camera pitch: 15° to 30° downward tilt).
- **Resolution:** 1080p (1920x1080) or 720p (1280x720).
- **Frame Rate:** 25 FPS or 30 FPS.
- **File Format:** H.264 MP4 or AVI.
- **Filename Convention:**
  `campus_<scenario>_<location>_<take_number>.mp4`
  Example: `campus_true_abandoned_lobby_take01.mp4`

---

## 4. Annotation Protocol

### A. Object Detection Annotations (YOLO Format)
For detector fine-tuning, extract frames and annotate using CVAT, Label Studio, or Roboflow:
- Classes:
  - `0`: `person`
  - `1`: `bag` (backpack, handbag, suitcase, box)
- Format per line: `<class_id> <x_center> <y_center> <width> <height>` (normalized coordinates [0..1]).

### B. Event-Level Ground Truth Annotations (JSON Format)
Save event ground truth in `data/local/annotations/campus_events.json`:
```json
{
  "campus_true_abandoned_take01.mp4": {
    "video": "campus_true_abandoned_take01.mp4",
    "duration_seconds": 35.0,
    "events": [
      {
        "object_id": "B01",
        "owner_id": "P01",
        "separation_start": 8.5,
        "abandoned_at": 23.5,
        "recovered_at": null,
        "event_type": "ABANDONED"
      }
    ]
  }
}
```
