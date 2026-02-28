import logging

LOGGER = logging.getLogger("cuda-bytetracker")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.propagate = False
