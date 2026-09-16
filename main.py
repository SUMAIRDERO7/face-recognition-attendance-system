#!/usr/bin/env python3
"""
main.py
=======

Terminal client for the Face Recognition Attendance System.

Usage:
    python main.py enroll --id 21AI001 --name "Sumair Dero" --photo face.jpg
    python main.py checkin --photo classroom.jpg
    python main.py report --date 2026-09-15
    python main.py people
    python main.py history --id 21AI001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.attendance_service import AttendanceService, EnrollmentError
from src.face_encoder import FaceEncoderError, build_default_encoder

FACES_DB = "data/faces.json"
ATTENDANCE_DB = "data/attendance.db"


def _load_image(path: str) -> np.ndarray:
    """Read an image file as RGB, exiting cleanly if it can't be read."""
    import cv2

    if not Path(path).exists():
        print(f"❌ Image not found: {path}", file=sys.stderr)
        sys.exit(1)

    bgr = cv2.imread(path)
    if bgr is None:
        print(f"❌ Could not read {path} as an image.", file=sys.stderr)
        sys.exit(1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _build_service() -> AttendanceService:
    encoder = build_default_encoder()
    if encoder.name != "dlib":
        print(
            "⚠️  Using the OpenCV fallback encoder — it compares faces on raw\n"
            "   appearance and is sensitive to lighting and pose. Install\n"
            "   `face_recognition` for production-grade accuracy.\n"
        )
    return AttendanceService.build(
        encoder=encoder, db_path=FACES_DB, attendance_db=ATTENDANCE_DB
    )


def cmd_enroll(service: AttendanceService, args: argparse.Namespace) -> int:
    image = _load_image(args.photo)
    try:
        count = service.enroll_from_image(args.id, args.name, image)
    except (EnrollmentError, FaceEncoderError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    service.save_database(FACES_DB)
    print(f"✅ Enrolled {args.name} ({args.id}) — {count} sample(s) on file.")
    print("   Tip: enroll 3-5 photos per person, in different lighting and poses.")
    return 0


def cmd_checkin(service: AttendanceService, args: argparse.Namespace) -> int:
    if len(service.database) == 0:
        print("❌ Nobody is enrolled yet. Run `enroll` first.", file=sys.stderr)
        return 1

    image = _load_image(args.photo)
    try:
        outcomes = service.process_frame(image)
    except FaceEncoderError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    if not outcomes:
        print("⚠️  No face detected in that image.")
        return 0

    print(f"Found {len(outcomes)} face(s):\n")
    for i, outcome in enumerate(outcomes, start=1):
        if not outcome.match.is_match:
            print(f"  {i}. Unknown — closest distance {outcome.match.distance:.3f}, nobody marked present")
        elif outcome.newly_recorded:
            print(
                f"  {i}. ✅ {outcome.match.name} checked in at "
                f"{outcome.record.check_in_time} ({outcome.record.status})"
            )
        else:
            print(f"  {i}. ℹ️  {outcome.match.name} was already recorded today")
    return 0


def cmd_report(service: AttendanceService, args: argparse.Namespace) -> int:
    report = service.daily_report(day=args.date)

    print(f"\nAttendance report — {report.date}")
    print("=" * 50)
    print(f"Enrolled: {report.total_enrolled}  |  Attendance rate: {report.attendance_rate:.0%}\n")

    for label, records in (("ON TIME", report.present), ("LATE", report.late)):
        print(f"{label} ({len(records)}):")
        for record in records:
            print(f"  {record.name:<25} {record.person_id:<12} in at {record.check_in_time}")
        if not records:
            print("  (none)")
        print()

    print(f"ABSENT ({len(report.absent)}):")
    for person_id, name in report.absent:
        print(f"  {name:<25} {person_id}")
    if not report.absent:
        print("  (none)")

    if args.csv:
        lines = ["person_id,name,status,check_in_time"]
        lines += [f"{r.person_id},{r.name},{r.status},{r.check_in_time}" for r in report.present + report.late]
        lines += [f"{pid},{name},absent," for pid, name in report.absent]
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.csv).write_text("\n".join(lines) + "\n")
        print(f"\n✅ CSV written to {args.csv}")
    return 0


def cmd_people(service: AttendanceService, args: argparse.Namespace) -> int:
    people = service.database.people
    if not people:
        print("Nobody enrolled yet.")
        return 0

    print(f"{len(people)} enrolled (encoder: {service.encoder.name}):\n")
    for person in people:
        print(f"  {person.name:<25} {person.person_id:<12} {person.sample_count} sample(s)")
    return 0


def cmd_history(service: AttendanceService, args: argparse.Namespace) -> int:
    records = service.log.history_for(args.id, limit=args.limit)
    if not records:
        print(f"No attendance history for {args.id}.")
        return 0

    print(f"Attendance history for {records[0].name} ({args.id}):\n")
    for record in records:
        print(f"  {record.date}  {record.check_in_time}  {record.status}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face recognition attendance system.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("enroll", help="Register a person's face.")
    p.add_argument("--id", required=True, help="Stable unique id (e.g. roll number).")
    p.add_argument("--name", required=True, help="Display name.")
    p.add_argument("--photo", required=True, help="Photo containing exactly one face.")

    p = sub.add_parser("checkin", help="Recognize faces in a photo and record attendance.")
    p.add_argument("--photo", required=True, help="Photo to check people in from.")

    p = sub.add_parser("report", help="Show a day's attendance report.")
    p.add_argument("--date", help="ISO date (YYYY-MM-DD). Defaults to today.")
    p.add_argument("--csv", help="Also write the report to this CSV path.")

    sub.add_parser("people", help="List everyone enrolled.")

    p = sub.add_parser("history", help="Show one person's attendance history.")
    p.add_argument("--id", required=True)
    p.add_argument("--limit", type=int, default=30)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        service = _build_service()
    except FaceEncoderError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    dispatch = {
        "enroll": cmd_enroll, "checkin": cmd_checkin,
        "report": cmd_report, "people": cmd_people, "history": cmd_history,
    }
    try:
        return dispatch[args.command](service, args)
    except (ValueError, FaceEncoderError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
