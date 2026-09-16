# 👤 Face Recognition Attendance System

> A production-oriented computer vision attendance system built with Python, OpenCV, SQLite, and Streamlit — designed around reliable recognition rules, database-level deduplication, testability, and responsible biometric-data handling.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python\&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-107%20Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit\&logoColor=white)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8?logo=opencv\&logoColor=white)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

The **Face Recognition Attendance System** is a computer vision application that allows organizations, classrooms, and controlled environments to enroll individuals and record attendance from images or camera input.

The project focuses not only on face recognition, but on the **software engineering rules surrounding recognition**:

* One attendance record per person per day
* Explicit rejection of unknown faces
* First-sighting-wins attendance policy
* Multi-sample face enrollment
* Database-enforced uniqueness
* Derived absence reporting
* Pluggable face-encoder architecture
* Streamlit dashboard and CLI interfaces
* Comprehensive automated testing
* Privacy and biometric-data considerations

The system supports two encoder implementations:

1. **dlib / `face_recognition`** — recommended for real recognition workloads
2. **OpenCV fallback encoder** — lightweight alternative for development and pipeline testing

The encoder is abstracted behind a protocol so the rest of the application does not depend directly on a particular face-recognition implementation.

---

## 🎯 Problem Statement

A basic face-recognition demo can identify a face.

A reliable attendance system has to answer much harder questions:

* What happens when the same person appears in multiple frames?
* What happens when an unknown person resembles an enrolled individual?
* Can a later detection overwrite an earlier attendance status?
* How are absences represented?
* What happens if an enrollment image contains multiple faces?
* How can the recognition backend be replaced without rewriting the application?
* Can the business logic be tested without installing heavyweight computer-vision dependencies?

This project addresses those problems by treating **recognition policy, data integrity, and testability as first-class engineering concerns**.

---

## ✨ Key Features

### 👤 Face Enrollment

* Enroll individuals using a unique ID and name
* Reject images containing zero faces
* Reject group photographs during enrollment
* Support multiple samples per person
* Maintain averaged face representations for matching

### 🔍 Face Recognition

* Detect faces using OpenCV
* Generate face representations through a pluggable encoder
* Compare detected faces against enrolled identities
* Apply a configurable recognition threshold
* Return `Unknown` when no enrolled identity is sufficiently close

### 🕐 Attendance Management

* One attendance record per person per day
* Database-level `UNIQUE(person_id, date)` constraint
* First-sighting-wins policy
* Prevent duplicate attendance records
* Distinguish between on-time and late attendance
* Generate daily attendance reports
* Derive absences instead of storing redundant absence records

### 🖥️ Interfaces

**Streamlit Dashboard**

* Enrollment interface
* Image upload
* Camera input
* Attendance processing
* Daily roster
* Attendance status
* Reporting

**CLI**

* Enroll users
* Process check-ins
* Generate reports
* List enrolled people
* View attendance history

### 🧪 Engineering Quality

* 107 automated tests
* 100% statement coverage across `src/`
* Dependency injection
* Protocol-based encoder abstraction
* SQLite constraints for data integrity
* Error handling
* Separation of concerns
* Testable service layer

---

# 🧠 Recognition Architecture

The system uses an abstraction layer around face encoders.

```text
                    +-----------------------+
                    |   AttendanceService   |
                    |        Facade         |
                    +-----------+-----------+
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
      +-------------+    +-------------+    +-------------+
      | FaceEncoder |    | FaceDatabase |    | Attendance  |
      |  Protocol   |    |             |    |    Log      |
      +------+------+    +-------------+    +-------------+
             |
       +-----+------+
       |            |
       v            v
+-------------+ +-------------+
| Dlib Encoder| | OpenCV      |
|             | | Encoder     |
+-------------+ +-------------+
| 128-D       | | 1024-D      |
| embeddings  | | appearance  |
+-------------+ +-------------+
```

### Why use a protocol?

The application should not care whether the underlying recognition engine is dlib, OpenCV, or another future implementation.

Instead:

```text
Application
     ↓
FaceEncoder Protocol
     ↓
Concrete Encoder
```

This provides:

* Replaceable recognition backends
* Easier testing
* Reduced coupling
* Cleaner architecture
* Better maintainability
* A straightforward path for future model upgrades

---

# 🔬 Encoder Backends

| Feature             | dlib / `face_recognition`          | OpenCV Fallback                               |
| ------------------- | ---------------------------------- | --------------------------------------------- |
| Representation      | 128-dimensional embedding          | 1024-dimensional normalized appearance vector |
| Backend             | dlib / ResNet-based model          | OpenCV                                        |
| Recognition quality | Intended for practical recognition | Development / demonstration                   |
| Lighting robustness | Higher                             | Limited                                       |
| Pose robustness     | Higher                             | Limited                                       |
| Installation        | Requires native compilation        | Lightweight                                   |
| Recommended use     | Real recognition workloads         | Development and testing                       |

### ⚠️ Important Accuracy Note

The OpenCV fallback is intentionally documented as a **development/demo backend**, not as an equivalent replacement for a learned face-embedding model.

The system therefore exposes the encoder choice rather than hiding the trade-off.

For real recognition workloads, the dlib-based backend should be used and evaluated against the intended deployment environment.

---

# 🛡️ Attendance Integrity Rules

The most important part of the system is not simply detecting a face — it is preventing incorrect attendance records.

## 1. One Record Per Person Per Day

Attendance uniqueness is enforced at the database level:

```sql
UNIQUE(person_id, date)
```

This means the database itself prevents duplicate attendance records.

Instead of relying exclusively on:

```text
if already_exists():
    ...
```

the application uses a schema-level constraint as the final integrity boundary.

---

## 2. First Sighting Wins

Once a person's attendance has been recorded for a day, subsequent detections do not replace the original record.

Example:

```text
08:30 → Present
09:45 → Late
```

The original `08:30` record remains unchanged.

This prevents repeated camera detections from accidentally changing an earlier attendance state.

---

## 3. Unknown Is a Valid Result

The system does **not** automatically assign the nearest enrolled identity.

Instead:

```text
Detected Face
      ↓
Compare Against Enrolled Faces
      ↓
Distance ≤ Threshold?
      ├── YES → Recognized Person
      └── NO  → Unknown
```

This distinction is critical.

A recognition system that always chooses the closest enrolled person can transform uncertainty into an incorrect identity.

---

## 4. Absence Is Derived

The database stores attendance events, not absence events.

For a given date:

```text
Enrolled People
       -
People With Attendance Records
       =
Absent People
```

This avoids maintaining two competing sources of truth.

---

## 5. Enrollment Requires Exactly One Face

Enrollment images must contain exactly one detectable face.

```text
0 faces  → Reject
1 face   → Accept
2+ faces → Reject
```

This prevents accidentally associating the wrong person with an enrollment record.

---

## 6. Multiple Samples Improve Representation

A single image represents one pose and lighting condition.

The system supports multiple samples for each person and computes a representative mean embedding/vector.

For example:

```text
Sample 1 ─┐
Sample 2 ─┼──→ Representative Face Representation
Sample 3 ─┘
```

Using several enrollment samples can provide a more useful representation across changes in pose and appearance.

---

# 🏗️ Architecture

```text
┌───────────────────────────────────────────────┐
│                 Presentation                  │
│                                               │
│     Streamlit UI             CLI              │
│       app.py               main.py            │
└───────────────┬───────────────────┬───────────┘
                │                   │
                └─────────┬─────────┘
                          ↓
              ┌──────────────────────┐
              │ AttendanceService    │
              │      Facade          │
              └──────────┬───────────┘
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
   FaceEncoder     FaceDatabase    AttendanceLog
      Protocol       Enrollment       SQLite
          │           Matching        Reporting
          │
     ┌────┴─────┐
     ↓          ↓
   dlib       OpenCV
  Encoder     Encoder
```

### Architectural Principles

* **Separation of concerns**
* **Dependency inversion**
* **Protocol-based abstraction**
* **Database-enforced integrity**
* **Testable business logic**
* **Replaceable infrastructure**
* **Thin presentation layer**

---

# 📁 Project Structure

```text
face-attendance/
│
├── src/
│   ├── face_encoder.py
│   ├── face_database.py
│   ├── attendance_log.py
│   └── attendance_service.py
│
├── tests/
│   ├── test_face_encoder.py
│   ├── test_face_database.py
│   ├── test_attendance_log.py
│   └── test_attendance_service.py
│
├── data/
│   ├── faces.json
│   └── attendance.db
│
├── app.py
├── main.py
├── requirements.txt
├── pytest.ini
├── .gitignore
├── GUIDE.txt
└── README.md
```

---

# ⚙️ Technology Stack

| Technology                  | Purpose                             |
| --------------------------- | ----------------------------------- |
| **Python 3.12**             | Core application                    |
| **OpenCV**                  | Face detection and image processing |
| **dlib / face_recognition** | Learned face representations        |
| **Streamlit**               | Interactive web interface           |
| **SQLite**                  | Attendance persistence              |
| **Pytest**                  | Automated testing                   |
| **pytest-cov**              | Coverage measurement                |
| **JSON**                    | Lightweight enrollment storage      |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/SUMAIRDERO7/face-attendance.git
cd face-attendance
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install the Recommended Recognition Backend

For practical face recognition:

```bash
pip install face_recognition
```

> `face_recognition` depends on dlib, which may require native C++ build tooling and can take significantly longer to install than typical Python packages.

---

# ▶️ Usage

## Streamlit Application

Start the dashboard:

```bash
streamlit run app.py
```

The application provides:

* Person enrollment
* Image-based check-in
* Camera-based check-in
* Attendance roster
* Daily status
* Attendance reporting

---

## CLI

### Enroll a Person

```bash
python main.py enroll \
    --id 21AI001 \
    --name "Sumair Dero" \
    --photo face.jpg
```

### Process Attendance

```bash
python main.py checkin --photo classroom.jpg
```

### Generate Daily Report

```bash
python main.py report \
    --date 2026-09-15 \
    --csv attendance.csv
```

### List Enrolled People

```bash
python main.py people
```

### View Attendance History

```bash
python main.py history --id 21AI001
```

---

# 🧪 Testing

Run the complete test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

### Current Test Results

```text
107 tests passing
100% statement coverage
```

Coverage:

```text
Name                         Stmts   Miss   Cover
------------------------------------------------
src/attendance_log.py           90      0   100%
src/attendance_service.py       59      0   100%
src/face_database.py            97      0   100%
src/face_encoder.py             93      0   100%
------------------------------------------------
TOTAL                           339      0   100%
```

---

# 🔬 Testing Strategy

The project separates **application correctness** from **third-party library behavior**.

### Encoder Dependency Injection

The dlib implementation is accessed through the encoder abstraction.

Tests can therefore inject a controlled implementation without requiring dlib to be installed.

This allows testing of:

* Encoder selection
* Input validation
* Error handling
* Face-location / encoding handling
* Matching behavior
* Service integration

without making the complete test suite dependent on a heavyweight native dependency.

### OpenCV Detection

The test suite does not pretend that OpenCV's Haar cascade will reliably detect synthetic images.

Instead, the project tests its own image-processing and business-logic branches, while the actual computer-vision pipeline is separately verified with a real end-to-end run.

This keeps the test suite focused on **code owned by the project rather than testing OpenCV itself**.

---

# ✅ End-to-End Verification

The complete pipeline was executed using the actual OpenCV face-detection pipeline and generated test imagery.

Example result:

```text
Enrolled Sumair Dero (21AI001) — 1 sample(s) on file.

Found 1 face(s):

1. Sumair Dero checked in at 17:37:05 (late)
```

Running the check-in again:

```text
1. Sumair Dero was already recorded today
```

The service-level verification also confirmed that:

* A later check-in does not overwrite an earlier attendance record.
* Duplicate attendance is prevented.
* An enrolled person who does not appear is derived as absent.
* Attendance rate is calculated from the resulting daily state.

---

# 🔐 Privacy & Responsible Use

Face recognition involves biometric information and should be treated accordingly.

This project is a portfolio and educational implementation and **should not be treated as a production biometric deployment without additional security, privacy, and accuracy evaluation**.

A real deployment should consider:

### Consent

Individuals should be informed about the system and provide appropriate consent where required.

### Data Protection

Face representations should receive appropriate:

* Encryption
* Access control
* Retention policies
* Deletion mechanisms
* Backup controls

### Liveness Detection

The current system does not provide robust anti-spoofing.

A photograph or screen displaying a face could potentially be presented to the camera.

### Accuracy Evaluation

Recognition performance should be evaluated using representative data and deployment-specific conditions rather than relying on a single accuracy number.

### Human Override

Attendance systems should provide a controlled mechanism for correcting incorrect recognition results.

---

# 🚀 Deployment

## Streamlit Community Cloud

The Streamlit application can be deployed for demonstration purposes.

However, the current local JSON and SQLite storage model is not appropriate for durable production persistence in an ephemeral deployment environment.

For a production architecture, consider:

```text
Streamlit / Web UI
        ↓
Application API
        ↓
PostgreSQL / Managed Database
        ↓
Secure Object Storage
```

with appropriate authentication and biometric-data protection.

## Local Kiosk

For a classroom or controlled environment, a local machine with a webcam can be a practical deployment model.

Advantages include:

* Local processing
* Lower latency
* Reduced network dependency
* Better control over biometric data
* Easier physical access control

---

# 🗺️ Roadmap — Version 2.0

The following features are **planned future work and are not currently implemented**:

* [ ] Liveness / anti-spoofing detection
* [ ] Live video stream processing
* [ ] Temporal face-track smoothing
* [ ] Improved modern face detector
* [ ] Encrypted biometric storage
* [ ] Data retention and deletion workflow
* [ ] Admin correction interface
* [ ] Audit trail
* [ ] Multi-camera support
* [ ] Role-based authentication
* [ ] PostgreSQL backend
* [ ] Recognition-performance evaluation
* [ ] False-positive / false-negative analysis
* [ ] Deployment-specific benchmarking

---

# 📊 Future Evaluation Metrics

A future research-oriented version should evaluate:

```text
Recognition Accuracy
Precision
Recall
F1 Score
False Acceptance Rate (FAR)
False Rejection Rate (FRR)
Inference Latency
Detection FPS
Enrollment Robustness
Lighting Robustness
Pose Robustness
```

This would move the project beyond a demonstration system toward a more rigorous computer-vision evaluation framework.

---

# 🧩 Engineering Highlights

This project demonstrates several software-engineering principles beyond basic computer vision:

### Dependency Inversion

The attendance service depends on an encoder abstraction rather than a concrete recognition library.

### Database Integrity

Business-critical uniqueness is enforced by the database.

### Separation of Concerns

Face processing, enrollment, attendance persistence, reporting, and presentation are separated.

### Testability

External dependencies are replaceable through dependency injection.

### Explicit Failure States

`Unknown` is represented explicitly rather than forcing every face into an enrolled identity.

### Single Source of Truth

Absence is derived from enrollment and attendance data rather than stored redundantly.

---

# 💡 What I Learned

Building this system reinforced an important engineering lesson:

> **A machine-learning model is only one component of an AI system.**

The reliability of the final application depends equally on:

```text
Model
  +
Data
  +
Business Rules
  +
Persistence
  +
Testing
  +
Error Handling
  +
Security
  +
Responsible Deployment
```

The project therefore focuses as much on **system design and correctness** as it does on computer vision.

---

# 🏷️ GitHub Topics

```text
python
computer-vision
face-recognition
opencv
dlib
attendance-system
streamlit
sqlite
pytest
biometrics
machine-learning
artificial-intelligence
```

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# 👨‍💻 Author

**Sumair Ahmed Dero**

BS Artificial Intelligence Student
University of Sindh, Jamshoro

Focused on:

* Artificial Intelligence
* Machine Learning
* Computer Vision
* Natural Language Processing
* AI Engineering
* Research

---

# ⭐ Support

If you find this project useful for learning or experimentation, consider giving the repository a ⭐ on GitHub.

---

## 📌 Project Summary

**Face Recognition Attendance System** is a computer-vision attendance application that combines face recognition with reliable attendance-management rules. Its architecture separates the recognition backend from application logic, enforces one-record-per-person-per-day at the database level, explicitly rejects uncertain identities, derives absences from attendance records, and maintains a comprehensive automated test suite.

The project demonstrates how a computer-vision model can be integrated into a maintainable, testable software system rather than treated as an isolated ML demo.
