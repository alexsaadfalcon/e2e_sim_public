"""A small Screenplay-pattern core for procedural, user-journey UI tests.

Why this instead of plain ``assert`` sequences: the Screenplay pattern (Actor ->
Tasks -> Interactions, plus Questions to check state) lets a test read as the exact
ordered script of what a *user* does, with each step named -- so a failure says
"Alice while 'validate the loaded scenario': <what went wrong>" instead of a bare
assertion error 12 lines into an opaque function. See tests/e2e_ui/README.md.

Vocabulary (kept intentionally tiny):
  * Actor        -- performs the journey; holds the Playwright ``page`` (its ability
                    to browse the web) and a name for readable failure messages.
  * Task         -- a business-meaningful step ("load the reference scenario"). Any
                    object with ``perform_as(actor)``. ``Task.where(name, fn)`` wraps
                    a plain function into a named Task.
  * Question     -- reads state ("the validation text"); ``answered_by(actor)``.
  * Ensure.that  -- a checkpoint: pairs a Question with a matcher and raises a clear
                    AssertionError if it does not hold. Put these ALONG the journey,
                    not only at the end, so failures localize to the step that broke.

Interactions (Click/Fill/... on the concrete app) live in ``_ui_app.py`` so every
locator sits in one place, per Page-Object practice.
"""

from __future__ import annotations

from typing import Any, Callable


class Actor:
    """Someone who drives the app through a scripted journey."""

    def __init__(self, name: str, page: Any):
        self.name = name
        self.page = page  # the "ability to browse the web"

    # -- performing tasks ----------------------------------------------------
    def attempts_to(self, *tasks: "Task") -> "Actor":
        """Perform each task in order; annotate any failure with actor + step name."""
        for task in tasks:
            try:
                task.perform_as(self)
            except Exception as e:  # noqa: BLE001 -- re-raise with journey context
                step = getattr(task, "name", task.__class__.__name__)
                raise AssertionError(
                    f"[{self.name}] while '{step}': {type(e).__name__}: {e}"
                ) from e
        return self

    # -- asking questions / checkpoints -------------------------------------
    def asks(self, question: "Question") -> Any:
        return question.answered_by(self)

    def should(self, *checks: "Ensure") -> "Actor":
        """Alias for attempts_to, read as assertions ('actor should see ...')."""
        return self.attempts_to(*checks)


class Task:
    """Base class for a named step. Subclass and implement ``perform_as``, or use
    ``Task.where(name, fn)`` to wrap a function."""

    name = "task"

    def perform_as(self, actor: Actor) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def where(name: str, fn: Callable[[Actor], None]) -> "Task":
        return _FnTask(name, fn)


class _FnTask(Task):
    def __init__(self, name: str, fn: Callable[[Actor], None]):
        self.name = name
        self._fn = fn

    def perform_as(self, actor: Actor) -> None:
        self._fn(actor)


class Question:
    """Reads a piece of state the journey wants to check."""

    name = "question"

    def __init__(self, name: str, fn: Callable[[Actor], Any]):
        self.name = name
        self._fn = fn

    def answered_by(self, actor: Actor) -> Any:
        return self._fn(actor)


# ------------------------------------------------------------------ matchers
# Matchers are (predicate, description) pairs; description feeds the failure text.

def contains(substr: str):
    return (lambda v: substr in (v or ""), f"contain {substr!r}")


def does_not_contain(substr: str):
    return (lambda v: substr not in (v or ""), f"not contain {substr!r}")


def equals(expected: Any):
    return (lambda v: v == expected, f"equal {expected!r}")


def is_true():
    return (lambda v: bool(v), "be truthy")


def at_least(n: int):
    return (lambda v: v is not None and v >= n, f"be >= {n}")


class Ensure(Task):
    """A checkpoint: assert a matcher holds for a Question's answer."""

    def __init__(self, question: Question, matcher, label: str = ""):
        self._q = question
        self._pred, self._desc = matcher
        self.name = label or f"ensure {question.name} {self._desc}"

    @staticmethod
    def that(question: Question, matcher, label: str = "") -> "Ensure":
        return Ensure(question, matcher, label)

    def perform_as(self, actor: Actor) -> None:
        value = self._q.answered_by(actor)
        if not self._pred(value):
            raise AssertionError(
                f"expected {self._q.name} to {self._desc}, but got {value!r}"
            )
