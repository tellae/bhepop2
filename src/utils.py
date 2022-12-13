"""
Utility functions
"""

import pandas as pd
import logging as lg

logger_level = lg.DEBUG
logger_name = "hepop2_logger"


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
        formatter = lg.Formatter("%(levelname)s :: %(message)s")
        stream_handler = lg.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logger_level)
        logger.handler_set = True

    return logger




def read_filosofi(
    path: str, sheet: str, code_insee: str, xls_file="FILO_DISP_COM.xls", skip_rows=5
):
    """
    Read Filosofi data from raw xls file

    :param path: full path to directory containing xls file
    :param sheet: sheet name where to get data
    :param code_insee: insee code of selected municipality
    :param xls_file: name of xls file
    :param skip_rows: number of rows to skip in sheet
    :return: data frame of data
    """
    data_frame = pd.read_excel(
        path + xls_file,
        sheet_name=sheet,
        skiprows=skip_rows,
    ).query("CODGEO=='" + code_insee + "'")

    return data_frame
