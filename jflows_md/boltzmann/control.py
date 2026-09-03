"""Manual rejection of the running stage attempt: Ctrl+C in its terminal.

A Boltzmann generator run is watched through its log; when a stage attempt
is visibly not converging, the user can reject it before its validation
instead of waiting for the ESS gate or killing the process. ``Manual_Reject``
installs a handler for ``SIGINT`` (Ctrl+C) that only raises a
flag; ``iterate_boltzmann`` consumes the flag once per attempt, after the
trainer returns, and treats the attempt as rejected: the endpoint shrinks and
the attempt is repeated, exactly as after a failed ESS gate. The trainer runs
its scan one compiled step at a time (``REJECT_CHECK_STEPS = 1``) and stops
at the first step boundary after the signal, so the request is received
within one gradient step; the pool, the weights, and the advance check it
between their chunks. The flag is off unless the caller passes the object as
``reject_requested``; a fixed schedule has nothing to shrink and ignores it.

    with Manual_Reject() as reject:            # Ctrl+C rejects the attempt
        samples, stages, seconds = boltzmann_forward_KLX_G(
            ..., reject_requested=reject,
        )

A second Ctrl+C within ``STOP_WINDOW_SECONDS`` stops the program as usual, so
the key keeps its ordinary meaning when pressed twice. For a detached run the
same effect is ``kill -INT <pid>``; the driver prints that command.
"""

import os
import signal
import time

from ..utils.control import Manual_Rejection


__all__ = ["Manual_Reject", "Manual_Rejection", "STOP_WINDOW_SECONDS"]

STOP_WINDOW_SECONDS = 3.0   # a second Ctrl+C within this window stops the program


class Manual_Reject:
    """Ctrl+C raises a rejection flag; a second Ctrl+C within the window stops the run."""

    def __init__(self, signum: int = signal.SIGINT):
        self.signum = signum
        self._requested = False
        self._previous = None
        self._raised_at = 0.0

    @property
    def command(self) -> str:
        """How to reject the running attempt: the key, or the signal for a detached run."""
        name = signal.Signals(self.signum).name.removeprefix("SIG")
        key = "Ctrl+C" if self.signum == signal.SIGINT else f"kill -{name} {os.getpid()}"
        return f"{key} (detached: kill -{name} {os.getpid()}); twice within {STOP_WINDOW_SECONDS:.0f}s stops the run"

    def __enter__(self) -> "Manual_Reject":
        self._previous = signal.signal(self.signum, self._handle)
        return self

    def __exit__(self, *exc) -> bool:
        signal.signal(self.signum, self._previous)
        return False

    def _handle(self, signum, frame) -> None:
        now = time.monotonic()
        if self._requested and now - self._raised_at < STOP_WINDOW_SECONDS:
            raise KeyboardInterrupt   # second Ctrl+C in the window: stop as usual
        self._requested, self._raised_at = True, now

    def peek(self) -> bool:
        """Read the flag without clearing it (the trainers stop their scan on it)."""
        return self._requested

    def __call__(self) -> bool:
        requested, self._requested = self._requested, False
        return requested
