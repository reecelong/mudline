#!/usr/bin/env python3
"""Smoke test — verify mudline works against a real encrypted iOS backup.

Targeted test: finds a specific contact's handles in AddressBook, then
extracts only their message thread from sms.db. Keeps memory usage low.

Usage:
    python smoke_test.py
    python smoke_test.py "Finnegan Long"
"""

from __future__ import annotations

import getpass
import sqlite3
import sys
from datetime import datetime, timedelta

from mudline.foundation.crypto import KeybagDecryptor
from mudline.foundation.discovery import BackupDiscovery
from mudline.foundation.manifest import ManifestResolver

COCOA_EPOCH = datetime(2001, 1, 1)
TARGET_CONTACT = "Finnegan Long"


def cocoa_ns_to_datetime(cocoa_ns: int) -> datetime:
    """Convert Cocoa epoch nanoseconds to datetime."""
    return COCOA_EPOCH + timedelta(seconds=cocoa_ns / 1e9)


def find_contact_handles(resolver: ManifestResolver, first: str, last: str) -> list[str]:
    """Look up a contact's phone numbers and emails from AddressBook."""
    contacts_path = resolver.resolve("HomeDomain", "Library/AddressBook/AddressBook.sqlitedb")
    conn = sqlite3.connect(f"file:{contacts_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # Find the person
        person = conn.execute(
            "SELECT ROWID FROM ABPerson WHERE First = ? AND Last = ?",
            (first, last),
        ).fetchone()
        if person is None:
            return []

        # Get their phone numbers and emails (property 3=phone, 4=email)
        rows = conn.execute(
            "SELECT value FROM ABMultiValue WHERE record_id = ? AND property IN (3, 4)",
            (person["ROWID"],),
        ).fetchall()
        return [r["value"] for r in rows]
    finally:
        conn.close()


def extract_messages_for_handles(
    resolver: ManifestResolver, handles: list[str], limit: int = 50
) -> list[dict]:
    """Extract recent messages matching the given handles."""
    sms_path = resolver.resolve("HomeDomain", "Library/SMS/sms.db")
    conn = sqlite3.connect(f"file:{sms_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # Map handle strings to handle ROWIDs
        placeholders = ",".join("?" * len(handles))
        handle_rows = conn.execute(
            f"SELECT ROWID, id FROM handle WHERE id IN ({placeholders})",
            handles,
        ).fetchall()

        if not handle_rows:
            # Try normalized lookup (strip +1 prefix, etc.)
            normalized = []
            for h in handles:
                normalized.append(h)
                # Add common variants
                if h.startswith("+1"):
                    normalized.append(h[2:])
                elif not h.startswith("+"):
                    normalized.append("+1" + h)
            placeholders = ",".join("?" * len(normalized))
            handle_rows = conn.execute(
                f"SELECT ROWID, id FROM handle WHERE id IN ({placeholders})",
                normalized,
            ).fetchall()

        if not handle_rows:
            return []

        handle_ids = [r["ROWID"] for r in handle_rows]
        handle_map = {r["ROWID"]: r["id"] for r in handle_rows}

        # Find chat IDs that include these handles
        placeholders = ",".join("?" * len(handle_ids))
        chat_rows = conn.execute(
            f"""
            SELECT DISTINCT cmj.chat_id
            FROM chat_message_join cmj
            JOIN message m ON m.ROWID = cmj.message_id
            WHERE m.handle_id IN ({placeholders})
            """,
            handle_ids,
        ).fetchall()
        chat_ids = [r["chat_id"] for r in chat_rows]

        if not chat_ids:
            return []

        # Pull recent messages from those chats
        placeholders = ",".join("?" * len(chat_ids))
        rows = conn.execute(
            f"""
            SELECT
                m.ROWID,
                m.text,
                m.handle_id,
                m.date,
                m.is_from_me,
                cmj.chat_id
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            WHERE cmj.chat_id IN ({placeholders})
                AND m.text IS NOT NULL
                AND m.text != ''
            ORDER BY m.date DESC
            LIMIT ?
            """,
            [*chat_ids, limit],
        ).fetchall()

        messages = []
        for row in rows:
            try:
                ts = cocoa_ns_to_datetime(row["date"])
            except (ValueError, OverflowError):
                ts = None

            messages.append({
                "id": row["ROWID"],
                "text": row["text"],
                "handle": handle_map.get(row["handle_id"], "<you>"),
                "is_from_me": bool(row["is_from_me"]),
                "timestamp": ts,
                "chat_id": row["chat_id"],
            })

        # Reverse so oldest first
        messages.reverse()
        return messages
    finally:
        conn.close()


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else TARGET_CONTACT
    parts = target.rsplit(" ", 1)
    if len(parts) != 2:
        print(f"Usage: {sys.argv[0]} 'First Last'")
        return 1
    first, last = parts

    # Step 1: Discover
    print("=== Discovering backups ===")
    backups = BackupDiscovery().discover()
    if not backups:
        print("No iOS backups found.")
        return 1

    backup = backups[0]
    print(f"Found: {backup.device_name} (iOS {backup.ios_version}, "
          f"encrypted={backup.is_encrypted})")

    # Step 2: Decrypt
    if not backup.is_encrypted:
        resolver = ManifestResolver(backup.path)
        decryptor = None
    else:
        print("\n=== Decrypting ===")
        password = getpass.getpass("Backup password: ")
        try:
            decryptor = KeybagDecryptor(backup.path, password)
        except Exception as e:
            print(f"Decryption failed: {e}")
            return 1

        resolver = ManifestResolver(backup.path, decryptor)

    print(f"Manifest: {resolver.count()} files across {len(resolver.list_domains())} domains")

    # Step 3: Find contact
    print(f"\n=== Looking up {first} {last} ===")
    handles = find_contact_handles(resolver, first, last)
    if not handles:
        print(f"Contact '{first} {last}' not found in AddressBook.")
        if decryptor:
            decryptor.close()
        return 1

    print(f"Found handles: {handles}")

    # Step 4: Extract messages
    print(f"\n=== Extracting messages (last 50) ===")
    messages = extract_messages_for_handles(resolver, handles, limit=50)
    if not messages:
        print("No messages found for those handles.")
    else:
        print(f"Found {len(messages)} messages:\n")
        for msg in messages:
            direction = "→" if msg["is_from_me"] else "←"
            ts = msg["timestamp"].strftime("%Y-%m-%d %H:%M") if msg["timestamp"] else "?"
            text = msg["text"][:120]
            print(f"  {ts} {direction} {text}")

    # Cleanup
    if decryptor:
        decryptor.close()

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
