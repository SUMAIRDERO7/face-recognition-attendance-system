"""
app.py
======

Streamlit UI for the Face Recognition Attendance System.

Layout is a **kiosk dashboard**: a live roster board on the left showing
who's checked in, who's late, and who's still missing, with the capture
panel on the right. This is a new shape versus every prior project
(Day 41 chat thread, Day 42 document library, Day 43 video theater,
Day 44 mode-tabbed playground, Day 45 audio lab, Day 46 step wizard,
Day 47 trace timeline).

The roster-first arrangement is deliberate: in a real deployment the
screen faces a room, and the question everyone actually wants answered
is "am I marked present yet?" — not "what does the camera see right
now". The recognition panel serves the roster, not the other way round.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.attendance_service import AttendanceService, EnrollmentError
from src.face_encoder import FaceEncoderError, build_default_encoder

st.set_page_config(page_title="Face Attendance", page_icon="👤", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --navy: #0A2540; --royal: #2563EB; --sky: #38BDF8;
        --success: #10B981; --warn: #F59E0B; --danger: #EF4444;
        --bg: #F8FAFC; --text-primary: #111827; --text-secondary: #4B5563;
    }
    .stApp { background: var(--bg); }
    .app-title { font-size: 1.6rem; font-weight: 800; color: var(--navy); margin-bottom: 0; }
    .app-subtitle { color: var(--text-secondary); font-size: 0.92rem; margin: 0.15rem 0 1rem; }

    .roster-row {
        display: flex; align-items: center; justify-content: space-between;
        background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid #E2E8F0;
        border-radius: 0 8px 8px 0; padding: 0.5rem 0.75rem; margin-bottom: 0.35rem;
    }
    .roster-row.present { border-left-color: var(--success); }
    .roster-row.late { border-left-color: var(--warn); }
    .roster-row.absent { border-left-color: #CBD5E1; opacity: 0.75; }
    .roster-name { font-weight: 600; color: var(--text-primary); font-size: 0.88rem; }
    .roster-meta { font-size: 0.74rem; color: var(--text-secondary); }
    .pill {
        border-radius: 999px; padding: 0.1rem 0.6rem; font-size: 0.68rem;
        font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    }
    .pill-present { background: #ECFDF5; color: var(--success); }
    .pill-late { background: #FFFBEB; color: var(--warn); }
    .pill-absent { background: #F1F5F9; color: var(--text-secondary); }

    .stat-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 0.75rem; text-align: center;
    }
    .stat-value { font-size: 1.6rem; font-weight: 800; line-height: 1; }
    .stat-label {
        font-size: 0.66rem; color: var(--text-secondary);
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.3rem;
    }
    .empty-state { text-align: center; padding: 2rem 1rem; color: var(--text-secondary); }
    .empty-state h4 { color: var(--navy); margin-bottom: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

FACES_DB = Path("data/faces.json")
ATTENDANCE_DB = Path("data/attendance.db")
BOX_COLOR_MATCH = (16, 185, 129)
BOX_COLOR_UNKNOWN = (239, 68, 68)


@st.cache_resource
def get_service() -> AttendanceService | None:
    try:
        encoder = build_default_encoder()
    except FaceEncoderError:
        return None
    return AttendanceService.build(
        encoder=encoder, db_path=FACES_DB, attendance_db=ATTENDANCE_DB
    )


def draw_outcomes(image: np.ndarray, outcomes) -> np.ndarray:
    """Draw labelled boxes around every recognized face."""
    annotated = image.copy()
    for outcome in outcomes:
        x, y, w, h = outcome.location.to_xywh()
        color = BOX_COLOR_MATCH if outcome.match.is_match else BOX_COLOR_UNKNOWN
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = outcome.label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(annotated, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


st.markdown('<p class="app-title">👤 Face Recognition Attendance</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Enroll faces once, then check people in from a photo. '
    "Each person is recorded at most once per day.</p>",
    unsafe_allow_html=True,
)

service = get_service()

if service is None:
    st.markdown(
        '<div class="empty-state"><h4>No face encoder available</h4>'
        "<p>Install OpenCV (and optionally <code>face_recognition</code>) to continue.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

with st.sidebar:
    st.subheader("System")
    encoder_name = service.encoder.name
    if encoder_name == "dlib":
        st.success("Using dlib embeddings (accurate)", icon="✅")
    else:
        st.warning(
            "Using the OpenCV fallback encoder. It compares faces on raw "
            "appearance, so it's sensitive to lighting and pose — fine for "
            "trying the system out, not for real attendance. Install "
            "`face_recognition` for production-grade accuracy.",
            icon="⚠️",
        )
    st.caption(f"Embedding size: {service.encoder.dimensions} · Threshold: {service.database.threshold}")

    st.divider()
    st.subheader("Enroll a person")
    with st.form("enroll", clear_on_submit=True):
        person_id = st.text_input("ID (e.g. roll number)")
        name = st.text_input("Full name")
        photo = st.file_uploader("Photo with exactly one face", type=["jpg", "jpeg", "png"])
        enrolled = st.form_submit_button("Enroll", use_container_width=True)

    if enrolled:
        if not person_id.strip() or not name.strip():
            st.error("ID and name are both required.", icon="⚠️")
        elif photo is None:
            st.error("Upload a photo to enroll from.", icon="⚠️")
        else:
            buffer = np.frombuffer(photo.getvalue(), dtype=np.uint8)
            bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                count = service.enroll_from_image(person_id, name, rgb)
                service.save_database(FACES_DB)
                st.success(f"{name} enrolled — {count} sample(s) on file.", icon="✅")
                st.rerun()
            except (EnrollmentError, FaceEncoderError) as exc:
                st.error(str(exc), icon="🚨")

    st.divider()
    if len(service.database) > 0:
        st.caption(f"{len(service.database)} enrolled")
        for person in service.database.people:
            cols = st.columns([3, 1])
            cols[0].markdown(
                f'<span class="roster-name">{person.name}</span>'
                f'<br><span class="roster-meta">{person.person_id} · '
                f'{person.sample_count} sample(s)</span>',
                unsafe_allow_html=True,
            )
            if cols[1].button("✕", key=f"rm_{person.person_id}"):
                service.database.remove(person.person_id)
                service.save_database(FACES_DB)
                st.rerun()
    else:
        st.caption("Nobody enrolled yet.")

roster_col, capture_col = st.columns([1, 1])

# ----------------------------------------------------------------- roster --

with roster_col:
    today = date.today().isoformat()
    st.markdown(f"### Roster — {today}")

    report = service.daily_report(day=today)

    s1, s2, s3 = st.columns(3)
    s1.markdown(
        f'<div class="stat-card"><div class="stat-value" style="color:#10B981">'
        f'{len(report.present)}</div><div class="stat-label">on time</div></div>',
        unsafe_allow_html=True,
    )
    s2.markdown(
        f'<div class="stat-card"><div class="stat-value" style="color:#F59E0B">'
        f'{len(report.late)}</div><div class="stat-label">late</div></div>',
        unsafe_allow_html=True,
    )
    s3.markdown(
        f'<div class="stat-card"><div class="stat-value" style="color:#4B5563">'
        f'{len(report.absent)}</div><div class="stat-label">absent</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    if report.total_enrolled == 0:
        st.markdown(
            '<div class="empty-state"><h4>Nobody enrolled</h4>'
            "<p>Add people in the sidebar to build the roster.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        for record in report.present + report.late:
            css = "present" if record.status == "present" else "late"
            st.markdown(
                f'<div class="roster-row {css}"><div>'
                f'<div class="roster-name">{record.name}</div>'
                f'<div class="roster-meta">{record.person_id} · in at {record.check_in_time}</div>'
                f'</div><span class="pill pill-{css}">{record.status}</span></div>',
                unsafe_allow_html=True,
            )
        for person_id, person_name in report.absent:
            st.markdown(
                f'<div class="roster-row absent"><div>'
                f'<div class="roster-name">{person_name}</div>'
                f'<div class="roster-meta">{person_id} · not seen today</div>'
                f'</div><span class="pill pill-absent">absent</span></div>',
                unsafe_allow_html=True,
            )

        st.download_button(
            "⬇️ Export today's roster (CSV)",
            data="person_id,name,status,check_in_time\n"
            + "\n".join(
                f"{r.person_id},{r.name},{r.status},{r.check_in_time}"
                for r in report.present + report.late
            )
            + "\n"
            + "\n".join(f"{pid},{nm},absent," for pid, nm in report.absent),
            file_name=f"attendance_{today}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------------------------------------------------------------- capture --

with capture_col:
    st.markdown("### Check in")

    if len(service.database) == 0:
        st.info("Enroll at least one person before checking anyone in.", icon="ℹ️")
    else:
        source = st.radio("Source", ["Upload photo", "Camera"], horizontal=True)
        captured = (
            st.camera_input("Take a photo")
            if source == "Camera"
            else st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="checkin")
        )

        if captured is not None:
            buffer = np.frombuffer(captured.getvalue(), dtype=np.uint8)
            bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            try:
                outcomes = service.process_frame(rgb)
            except FaceEncoderError as exc:
                st.error(str(exc), icon="🚨")
                outcomes = []

            if outcomes:
                st.image(draw_outcomes(rgb, outcomes), use_container_width=True)
                for outcome in outcomes:
                    if outcome.match.is_match and outcome.newly_recorded:
                        st.success(f"{outcome.match.name} checked in.", icon="✅")
                    elif outcome.match.is_match:
                        st.info(f"{outcome.match.name} was already recorded today.", icon="ℹ️")
                    else:
                        st.warning(
                            f"Unrecognized face (closest distance {outcome.match.distance:.3f}). "
                            "Nobody was marked present.",
                            icon="⚠️",
                        )
                st.rerun()
            else:
                st.warning("No face detected in that image.", icon="⚠️")
