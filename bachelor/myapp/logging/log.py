import logging
from functools import wraps
from inspect import signature

logging.basicConfig(level=logging.INFO, format="%(message)s")

def log_call(message: str, log_return: bool = False):
    """
    Decorator for string-based logging with argument placeholders.
    Example: @log_call("Processing user {user_id}", log_return=True)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            try:
                log_msg = message.format(**bound_args.arguments)
            except Exception as e:
                log_msg = f"{message} | (formatting failed: {e})"
            logging.info(log_msg)

            try:
                result = func(*args, **kwargs)
                if log_return:
                    logging.info("Return value: %r", result)
                return result
            except Exception as e:
                logging.exception("Exception in %s: %s", func.__name__, e)
                raise
        return wrapper
    return decorator