"""
db/controls.py — Read and write the engine_controls table (single row, id=1).

The engine checks this table on every cycle to determine whether the kill switch
is active, whether a close-all has been requested, and what position overrides
(if any) should override optimizer output.
"""

import json
from contextlib import contextmanager
from loguru import logger

from db.connection import get_conn, put_conn


# Safe defaults returned when the row is missing or the DB is unreachable.
_DEFAULTS: dict = {
    "id":                   1,
    "kill_switch":          False,
    "kill_mode":            "halt",
    "close_all_triggered":  False,
    "override_active":      False,
    "override_mode":        "weights",
    "manual_weights":       {},
    "manual_quantities":    {},
    "note":                 "",
    "updated_at":           None,
}


@contextmanager
def _db():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        put_conn(conn)


def load_controls() -> dict:
    """
    Read the engine_controls row where id = 1.

    Returns a dict with all columns. Falls back to _DEFAULTS on any error or
    if the row does not exist yet (table created but never seeded).
    """
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    id,
                    kill_switch,
                    kill_mode,
                    close_all_triggered,
                    override_active,
                    override_mode,
                    manual_weights,
                    manual_quantities,
                    note,
                    updated_at
                FROM engine_controls
                WHERE id = 1
            """)
            row = cur.fetchone()
            if row is None:
                logger.warning("engine_controls row not found — returning defaults")
                return dict(_DEFAULTS)

            (
                _id,
                kill_switch,
                kill_mode,
                close_all_triggered,
                override_active,
                override_mode,
                manual_weights,
                manual_quantities,
                note,
                updated_at,
            ) = row

            return {
                "id":                   _id,
                "kill_switch":          bool(kill_switch),
                "kill_mode":            kill_mode or "halt",
                "close_all_triggered":  bool(close_all_triggered),
                "override_active":      bool(override_active),
                "override_mode":        override_mode or "weights",
                "manual_weights":       manual_weights if isinstance(manual_weights, dict) else {},
                "manual_quantities":    manual_quantities if isinstance(manual_quantities, dict) else {},
                "note":                 note or "",
                "updated_at":           str(updated_at) if updated_at is not None else None,
            }

    except Exception as e:
        logger.error(f"load_controls failed: {e}")
        return dict(_DEFAULTS)


def save_controls(
    kill_switch: bool | None = None,
    kill_mode: str | None = None,
    close_all_triggered: bool | None = None,
    override_active: bool | None = None,
    override_mode: str | None = None,
    manual_weights: dict | None = None,
    manual_quantities: dict | None = None,
    note: str | None = None,
) -> None:
    """
    Upsert the engine_controls row (id = 1).

    Only kwargs that are not None are written; everything else is left unchanged.
    updated_at is always refreshed to NOW().

    Raises on DB error so the caller knows the write failed.
    """
    # Build the SET clause dynamically from provided (non-None) kwargs.
    updates: list[str] = ["updated_at = NOW()"]
    values: list = []

    if kill_switch is not None:
        updates.append("kill_switch = %s")
        values.append(bool(kill_switch))

    if kill_mode is not None:
        updates.append("kill_mode = %s")
        values.append(kill_mode)

    if close_all_triggered is not None:
        updates.append("close_all_triggered = %s")
        values.append(bool(close_all_triggered))

    if override_active is not None:
        updates.append("override_active = %s")
        values.append(bool(override_active))

    if override_mode is not None:
        updates.append("override_mode = %s")
        values.append(override_mode)

    if manual_weights is not None:
        updates.append("manual_weights = %s")
        values.append(json.dumps(manual_weights))

    if manual_quantities is not None:
        updates.append("manual_quantities = %s")
        values.append(json.dumps(manual_quantities))

    if note is not None:
        updates.append("note = %s")
        values.append(note)

    set_clause = ", ".join(updates)

    try:
        with _db() as conn:
            cur = conn.cursor()
            # Use an INSERT … ON CONFLICT upsert so the row is created if missing.
            # We only write the columns the caller supplied; the rest default to
            # the column DEFAULTs on INSERT or are left unchanged on UPDATE.
            cur.execute(
                f"""
                INSERT INTO engine_controls (id)
                VALUES (1)
                ON CONFLICT (id) DO UPDATE
                    SET {set_clause}
                """,
                values,
            )
            logger.info(f"engine_controls updated: {set_clause} | values={values}")

    except Exception as e:
        logger.error(f"save_controls failed: {e}")
        raise


def acknowledge_close_all() -> None:
    """
    Clear the close_all_triggered flag after the engine has processed it.
    Called by the live engine immediately after it has queued all close orders.
    """
    try:
        save_controls(close_all_triggered=False)
        logger.info("engine_controls: close_all_triggered acknowledged (reset to False)")
    except Exception as e:
        logger.error(f"acknowledge_close_all failed: {e}")
        raise
