import functools
import logging
import time
from collections.abc import Callable
from typing import TypeVar, cast

logger = logging.getLogger(__name__)

# Build retryable exception tuple — include psycopg if installed
_RETRYABLE: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, OSError)
try:
    import psycopg

    _RETRYABLE = (*_RETRYABLE, psycopg.OperationalError)
except ImportError:
    pass


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = _RETRYABLE,
):
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds (doubles each retry)
        exceptions: Exception types to retry on
    """

    F = TypeVar("F", bound=Callable[..., object])

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exception: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            fn.__name__,
                            max_attempts,
                            e,
                        )
                        raise

                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.1fs",
                        fn.__name__,
                        attempt,
                        max_attempts,
                        e,
                        delay,
                    )
                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            raise RuntimeError("retry loop exited without returning") from last_exception

        return cast(F, wrapper)

    return decorator
