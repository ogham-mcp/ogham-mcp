"""Debounced per-key scheduler for topic-summary recomputes (T1.4 Phase 5).

Hooks in Phase 6 enqueue one recompute per tag on every memory write.
Without debounce, a burst of 10 related writes would trigger 10 LLM
synthesizes on the same topic -- costly and redundant. The debouncer
coalesces arrivals within a configurable window (default 60s) so rapid-
fire ingest bursts collapse to a single recompile at the end of the
burst.

Design:
  * threading.Timer per (profile, topic_key). On re-enqueue the existing
    timer is cancelled and a fresh one started, so "last arrival wins"
    -- the recompile uses the full source set at fire time.
  * When a timer fires it submits the actual recompute_topic_summary
    call to a ThreadPoolExecutor so the LLM round-trip doesn't block
    the Timer thread (which would delay *other* queues firing).
  * flush() cancels pending timers and immediately runs them, then
    waits for the pool. Used by tests and at-shutdown.
  * Exceptions in recompute are logged, never raised -- one bad topic
    must not poison the queue for others.

Scope notes:
  * Per-key state is a process-wide dict. Multi-process deployments
    (gunicorn workers, etc) get per-process debouncing -- fine for
    v0.11.1 single-process MCP server, will need Redis-backed cursor
    if we multi-tenant-cluster this.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_DEBOUNCE_SECONDS = float(os.environ.get("OGHAM_SUMMARY_DEBOUNCE_SECONDS", "60"))
_MAX_WORKERS = int(os.environ.get("OGHAM_SUMMARY_WORKERS", "2"))

_pending_timers: dict[tuple[str, str], threading.Timer] = {}
_pending_lock = threading.Lock()

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()

_running_futures: set[Future] = set()
_running_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS,
                thread_name_prefix="ogham-summary",
            )
        return _executor


def _run_recompute(profile: str, topic_key: str, **kwargs: Any) -> Any:
    """Forward to the real recompute. Separated for test patchability.

    Tests patch this symbol to avoid pulling the whole DB/LLM stack into
    executor-only tests. The real impl imports lazily so the executor
    module stays importable in contexts that never recompute (benchmarks,
    schema-only tests).
    """
    from ogham.recompute import recompute_topic_summary

    return recompute_topic_summary(profile, topic_key, **kwargs)


def _invoke(profile: str, topic_key: str) -> None:
    """Runs inside the ThreadPoolExecutor thread. Catch+log exceptions."""
    try:
        _run_recompute(profile, topic_key)
    except Exception:
        logger.warning(
            "recompute %s/%s failed in summary executor",
            profile,
            topic_key,
            exc_info=True,
        )


def _fire(key: tuple[str, str]) -> None:
    """Called by threading.Timer when the debounce window expires.

    Hands the actual recompute off to the thread pool so the Timer
    thread is freed to fire other queues without serializing on one
    long LLM call.
    """
    with _pending_lock:
        _pending_timers.pop(key, None)
    profile, topic_key = key
    fut = _get_executor().submit(_invoke, profile, topic_key)
    with _running_lock:
        _running_futures.add(fut)
    fut.add_done_callback(_remove_future)


def _remove_future(fut: Future) -> None:
    with _running_lock:
        _running_futures.discard(fut)


def enqueue_for_tags(profile: str, tags: list[str] | None) -> int:
    """Enqueue a recompute for each tag on a memory mutation.

    Used by Phase 6 hooks (store / update / delete / reinforce / contradict).
    Safe to call with None or an empty list -- returns 0 and does nothing.
    Returns the number of tags enqueued so the caller can log if desired.
    """
    n = 0
    for tag in tags or []:
        enqueue(profile, tag)
        n += 1
    return n


def enqueue(profile: str, topic_key: str, *, debounce: float | None = None) -> None:
    """Schedule a topic-summary recompute, coalescing with any pending
    request for the same (profile, topic_key).

    Callers can override `debounce` for tests; production callers leave
    it None to use OGHAM_SUMMARY_DEBOUNCE_SECONDS (default 60.0).
    """
    window = _DEFAULT_DEBOUNCE_SECONDS if debounce is None else float(debounce)
    key = (profile, topic_key)
    with _pending_lock:
        existing = _pending_timers.get(key)
        if existing is not None:
            existing.cancel()
        timer = threading.Timer(window, _fire, args=(key,))
        timer.daemon = True
        _pending_timers[key] = timer
        timer.start()


def _drain_pending_now() -> list[Callable[[], None]]:
    """Pull all pending timers, cancel them, return fire-now callables.

    Called by flush() and shutdown() to bypass the debounce wait.
    Returns the list of deferred fires so the caller can decide whether
    to run them or discard them.
    """
    with _pending_lock:
        keys = list(_pending_timers.keys())
        for k in keys:
            t = _pending_timers.pop(k)
            t.cancel()
    return [(lambda k=k: _fire(k)) for k in keys]


def flush(timeout: float = 30.0) -> int:
    """Drain pending timers and wait for all running recomputes.

    Returns the number of recomputes dispatched by this flush. Does NOT
    cancel in-flight recomputes -- they run to completion subject to
    the timeout.
    """
    # 1. Cancel pending timers, submit their fires immediately.
    fires = _drain_pending_now()
    for f in fires:
        f()

    # 2. Wait for all running futures.
    with _running_lock:
        running_snapshot = list(_running_futures)
    for fut in running_snapshot:
        try:
            fut.result(timeout=timeout)
        except Exception:
            logger.warning("summary executor future failed during flush", exc_info=True)

    return len(fires)


def shutdown() -> None:
    """Best-effort shutdown: cancel pending timers, wait briefly for running.

    Registered via atexit so the process doesn't leave zombie Timers
    when the MCP server shuts down.
    """
    # Cancel pending without firing -- at shutdown we accept losing the
    # debounced-but-not-yet-fired recomputes. The nightly sweep + next
    # session's hash check recover them anyway.
    with _pending_lock:
        for t in _pending_timers.values():
            t.cancel()
        _pending_timers.clear()

    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=True, cancel_futures=False)
            _executor = None

    with _running_lock:
        _running_futures.clear()


atexit.register(shutdown)
