"""
Utility functions
"""

import logging as lg

# log utils (see logging library)

#: logging level
logger_level = lg.WARNING

#: logger name
logger_name = "bhepop2_logger"


class Bhepop2Logger:
    """
    Logging helper class for Bhepop2.

    See logging library for more information.
    """

    def __init__(self, default_log_level=lg.DEBUG):
        """
        Get the bhepop2 logger and store a default log level.

        :param default_log_level: default level of logged messages
        """

        self._logger = _get_logger()
        self._default_log_level = default_log_level

    def log(self, message: str, level: int = None):
        """
        Log a message using the package logger.

        :param message: message to be logged
        :param level: logging level
        """

        if level is None:
            level = self._default_log_level

        self._logger.log(level, message)


def log(message, level):
    """
    Log a message using the logging library.

    :param message: message to log
    :param level: logging level
    """

    # get logger object
    logger = _get_logger()

    # log message
    logger.log(level, message)


def _get_logger():
    """
    Create a logger or return the current one if already instantiated.

    :return: logging.logger
    """

    logger = lg.getLogger(logger_name)

    # if a logger with this name is not already set up
    if not getattr(logger, "handler_set", None):
        logger.propagate = False
        formatter = lg.Formatter("%(message)s")
        stream_handler = lg.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logger_level)
        logger.handler_set = True

    return logger


# json schema utils


# deprecated
# def add_defaults_and_validate_against_schema(instance, schema):
#     """
#     Add default values then validate instance against the schema.
#
#     :param instance: data instance
#     :param schema: json schema
#
#     :return: result data (copy of instance)
#     """
#
#     result = instance.copy()
#
#     # superficial default values set
#     for key in schema["properties"]:
#         if "default" in schema["properties"][key] and not key in result:
#             result[key] = schema["properties"][key]["default"]
#
#     # validate instance against schema
#     try:
#         validate(result, schema)
#     except ValidationError as e:
#         msg = e.message
#         raise ValueError(msg) from None
#
#     return result
