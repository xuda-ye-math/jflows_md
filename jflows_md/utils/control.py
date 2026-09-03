"""Rejection check shared by the long pre-scan steps (quench and temper, population MALA)."""

__all__ = ["Manual_Rejection", "check_rejection"]


class Manual_Rejection(Exception):
    """Raised inside a chunked computation when the manual-rejection flag is up."""


def check_rejection(reject_requested) -> None:
    """Raise ``Manual_Rejection`` if ``reject_requested.peek()`` is true (no-op for ``None``)."""
    if reject_requested is not None and reject_requested.peek():
        raise Manual_Rejection()
