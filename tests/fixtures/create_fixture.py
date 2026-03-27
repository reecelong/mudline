"""Generate a synthetic iOS backup fixture for testing all extractors.

Run this script to create a minimal but realistic backup structure in tests/fixtures/backup/.
All extractors should be tested against this fixture.

Usage:
    python tests/fixtures/create_fixture.py
"""

import hashlib
import os
import plistlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "backup"
COCOA_EPOCH = datetime(2001, 1, 1)


def cocoa_timestamp(dt: datetime) -> float:
    """Convert datetime to Cocoa epoch seconds."""
    return (dt - COCOA_EPOCH).total_seconds()


def file_hash(domain: str, relative_path: str) -> str:
    """Generate SHA-1 hash matching Apple's backup format."""
    key = f"{domain}-{relative_path}"
    return hashlib.sha1(key.encode()).hexdigest()


def store_file(backup_dir: Path, fileid: str, content: bytes) -> None:
    """Store a file in the 2-char prefix subdirectory structure."""
    subdir = backup_dir / fileid[:2]
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / fileid).write_bytes(content)


def create_manifest_db(backup_dir: Path, files: list[tuple[str, str, str, int]]) -> None:
    """Create Manifest.db with file mappings.

    Args:
        files: list of (fileID, domain, relativePath, flags) tuples
    """
    db_path = backup_dir / "Manifest.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE Files (
            fileID TEXT PRIMARY KEY,
            domain TEXT,
            relativePath TEXT,
            flags INTEGER,
            file BLOB
        )
    """)
    conn.executemany(
        "INSERT INTO Files (fileID, domain, relativePath, flags, file) VALUES (?, ?, ?, ?, ?)",
        [(fid, domain, rpath, flags, b"") for fid, domain, rpath, flags in files],
    )
    conn.commit()
    conn.close()


def create_info_plist(backup_dir: Path) -> None:
    """Create Info.plist with device metadata."""
    info = {
        "Device Name": "Test iPhone",
        "Display Name": "Test iPhone",
        "ICCID": "00000000000000000000",
        "IMEI": "000000000000000",
        "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
        "Phone Number": "+15551234567",
        "Product Name": "iPhone15,2",
        "Product Type": "iPhone15,2",
        "Product Version": "18.3",
        "Serial Number": "XXXXXXXXXXXX",
        "Target Identifier": "0000000000000000000000000000000000000000",
        "Unique Identifier": "abcdef1234567890abcdef1234567890abcdef12",
    }
    with open(backup_dir / "Info.plist", "wb") as f:
        plistlib.dump(info, f)


def create_manifest_plist(backup_dir: Path, encrypted: bool = False) -> None:
    """Create Manifest.plist."""
    manifest = {
        "BackupKeyBag": b"placeholder",
        "IsEncrypted": encrypted,
        "Version": "10.0",
        "Date": datetime(2026, 3, 15, 10, 30, 0),
    }
    with open(backup_dir / "Manifest.plist", "wb") as f:
        plistlib.dump(manifest, f)


def create_status_plist(backup_dir: Path) -> None:
    """Create Status.plist."""
    status = {
        "BackupState": "new",
        "Date": datetime(2026, 3, 15, 10, 30, 0),
        "IsFullBackup": True,
        "SnapshotState": "finished",
        "UUID": "ABCDEF12-3456-7890-ABCD-EF1234567890",
        "Version": "3.3",
    }
    with open(backup_dir / "Status.plist", "wb") as f:
        plistlib.dump(status, f)


def create_sms_db() -> bytes:
    """Create a synthetic sms.db with test messages."""
    import tempfile

    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)

    conn.executescript("""
        CREATE TABLE handle (
            ROWID INTEGER PRIMARY KEY,
            id TEXT UNIQUE,
            country TEXT,
            service TEXT
        );
        CREATE TABLE chat (
            ROWID INTEGER PRIMARY KEY,
            display_name TEXT,
            chat_identifier TEXT
        );
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            text TEXT,
            handle_id INTEGER,
            date INTEGER,
            is_from_me INTEGER,
            cache_has_attachments INTEGER DEFAULT 0
        );
        CREATE TABLE chat_message_join (
            chat_id INTEGER,
            message_id INTEGER
        );
        CREATE TABLE attachment (
            ROWID INTEGER PRIMARY KEY,
            filename TEXT,
            mime_type TEXT,
            total_bytes INTEGER
        );
        CREATE TABLE message_attachment_join (
            message_id INTEGER,
            attachment_id INTEGER
        );
    """)

    # Handles (contacts)
    handles = [
        (1, "+15551234567", "us", "iMessage"),
        (2, "+15559876543", "us", "iMessage"),
        (3, "sarah@example.com", "us", "iMessage"),
    ]
    conn.executemany("INSERT INTO handle VALUES (?, ?, ?, ?)", handles)

    # Chats
    chats = [
        (1, "", "+15551234567"),
        (2, "Family Group", "chat123456"),
        (3, "", "sarah@example.com"),
    ]
    conn.executemany("INSERT INTO chat VALUES (?, ?, ?)", chats)

    # Messages — using Cocoa timestamps
    base_time = datetime(2026, 2, 15, 14, 0, 0)
    t0 = cocoa_timestamp(base_time)
    t2 = cocoa_timestamp(base_time + timedelta(minutes=2))
    t5 = cocoa_timestamp(base_time + timedelta(minutes=5))
    t7 = cocoa_timestamp(base_time + timedelta(minutes=7))
    t3h = cocoa_timestamp(base_time + timedelta(hours=3))
    t1d = cocoa_timestamp(base_time + timedelta(days=1))
    messages = [
        (1, "Hey, did you call the plumber?", 3, int(t0 * 1e9), 0, 0),
        (2, "Yeah, he said he can come Thursday", 3, int(t2 * 1e9), 1, 0),
        (3, "The quote was $350 for the whole job", 3, int(t5 * 1e9), 0, 0),
        (4, "That sounds reasonable, let's go with it", 3, int(t7 * 1e9), 1, 0),
        (5, "Don't forget dinner tomorrow!", 1, int(t3h * 1e9), 0, 0),
        (6, "Happy birthday!! 🎂", 2, int(t1d * 1e9), 1, 0),
    ]
    conn.executemany("INSERT INTO message VALUES (?, ?, ?, ?, ?, ?)", messages)

    # Chat-message joins
    chat_msgs = [
        (3, 1), (3, 2), (3, 3), (3, 4),  # Sarah conversation
        (1, 5),  # Direct message
        (2, 6),  # Family group
    ]
    conn.executemany("INSERT INTO chat_message_join VALUES (?, ?)", chat_msgs)

    conn.commit()
    data = Path(db_path).read_bytes()
    conn.close()
    os.unlink(db_path)
    return data


def create_contacts_db() -> bytes:
    """Create a synthetic AddressBook.sqlitedb."""
    import tempfile

    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)

    conn.executescript("""
        CREATE TABLE ABPerson (
            ROWID INTEGER PRIMARY KEY,
            First TEXT,
            Last TEXT,
            Organization TEXT,
            Note TEXT
        );
        CREATE TABLE ABMultiValue (
            UID INTEGER PRIMARY KEY,
            record_id INTEGER,
            property INTEGER,
            identifier INTEGER,
            label TEXT,
            value TEXT
        );
    """)

    # property constants: 3=phone, 4=email
    persons = [
        (1, "Sarah", "Johnson", "Acme Corp", None),
        (2, "John", "Smith", None, "Plumber referral"),
        (3, "Mom", None, None, None),
    ]
    conn.executemany("INSERT INTO ABPerson VALUES (?, ?, ?, ?, ?)", persons)

    multi_values = [
        (1, 1, 3, 0, "_$!<Mobile>!$_", "+15559876543"),
        (2, 1, 4, 0, "_$!<Work>!$_", "sarah@example.com"),
        (3, 2, 3, 0, "_$!<Mobile>!$_", "+15551112222"),
        (4, 3, 3, 0, "_$!<Mobile>!$_", "+15551234567"),
    ]
    conn.executemany("INSERT INTO ABMultiValue VALUES (?, ?, ?, ?, ?, ?)", multi_values)

    conn.commit()
    data = Path(db_path).read_bytes()
    conn.close()
    os.unlink(db_path)
    return data


def create_call_history_db() -> bytes:
    """Create a synthetic CallHistory.storedata."""
    import tempfile

    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)

    conn.executescript("""
        CREATE TABLE ZCALLRECORD (
            Z_PK INTEGER PRIMARY KEY,
            ZADDRESS TEXT,
            ZDURATION REAL,
            ZDATE REAL,
            ZORIGINATED INTEGER,
            ZANSWERED INTEGER
        );
    """)

    base_time = datetime(2026, 3, 10, 9, 0, 0)
    ct0 = cocoa_timestamp(base_time)
    ct2h = cocoa_timestamp(base_time + timedelta(hours=2))
    ct1d = cocoa_timestamp(base_time + timedelta(days=1))
    calls = [
        (1, "+15559876543", 185.0, ct0, 1, 1),   # outgoing answered
        (2, "+15551234567", 0.0, ct2h, 0, 0),     # incoming missed
        (3, "+15551112222", 42.0, ct1d, 0, 1),    # incoming answered
    ]
    conn.executemany("INSERT INTO ZCALLRECORD VALUES (?, ?, ?, ?, ?, ?)", calls)

    conn.commit()
    data = Path(db_path).read_bytes()
    conn.close()
    os.unlink(db_path)
    return data


def main() -> None:
    """Generate the complete test fixture."""
    if FIXTURE_DIR.exists():
        import shutil
        shutil.rmtree(FIXTURE_DIR)

    FIXTURE_DIR.mkdir(parents=True)

    # Track all files for Manifest.db
    manifest_files: list[tuple[str, str, str, int]] = []

    # --- SMS database ---
    sms_domain = "HomeDomain"
    sms_path = "Library/SMS/sms.db"
    sms_hash = file_hash(sms_domain, sms_path)
    store_file(FIXTURE_DIR, sms_hash, create_sms_db())
    manifest_files.append((sms_hash, sms_domain, sms_path, 1))

    # --- Contacts database ---
    contacts_domain = "HomeDomain"
    contacts_path = "Library/AddressBook/AddressBook.sqlitedb"
    contacts_hash = file_hash(contacts_domain, contacts_path)
    store_file(FIXTURE_DIR, contacts_hash, create_contacts_db())
    manifest_files.append((contacts_hash, contacts_domain, contacts_path, 1))

    # --- Call History ---
    calls_domain = "HomeDomain"
    calls_path = "Library/CallHistoryDB/CallHistory.storedata"
    calls_hash = file_hash(calls_domain, calls_path)
    store_file(FIXTURE_DIR, calls_hash, create_call_history_db())
    manifest_files.append((calls_hash, calls_domain, calls_path, 1))

    # --- Plists ---
    create_info_plist(FIXTURE_DIR)
    create_manifest_plist(FIXTURE_DIR)
    create_status_plist(FIXTURE_DIR)
    create_manifest_db(FIXTURE_DIR, manifest_files)

    print(f"✅ Fixture created at {FIXTURE_DIR}")
    print(f"   {len(manifest_files)} data files")
    print(f"   Domains: {sorted(set(f[1] for f in manifest_files))}")


if __name__ == "__main__":
    main()
