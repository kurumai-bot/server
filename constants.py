from datetime import datetime
import logging
import os
from os import path
import time

from ai_interface import AIInterface
from db import Database


LOGGER_FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
logging.Formatter.converter = time.gmtime

LOGGER = logging.getLogger("kurumai")
LOGGER.propagate = False
LOGGER.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(LOGGER_FORMATTER)
stream_handler.setLevel(logging.INFO)
LOGGER.addHandler(stream_handler)

if not path.isdir("logs"):
    os.mkdir("logs")
file_handler = logging.FileHandler(
    path.join("logs", datetime.now().strftime("%d-%m-%Y--%H-%M-%S") + ".log")
)
file_handler.setFormatter(LOGGER_FORMATTER)
file_handler.setLevel(logging.DEBUG)
LOGGER.addHandler(file_handler)


HOST = "LOCALHOST"
PORT = 8080


SERVICE_HOST = "LOCALHOST"
SERVICE_PORT = 8079


with open("secrets", "r", encoding="utf-8") as secrets_file:
    SERVICE_AUTH_KEY = secrets_file.readline().removesuffix("\n").encode()
    SECRET_KEY = secrets_file.readline().removesuffix("\n")
    DB_URL = secrets_file.readline().removesuffix("\n")


DB = Database(DB_URL, logger=LOGGER.getChild("db"), cache_logger=LOGGER.getChild("db.cache"))


AIINTERFACE = AIInterface(SERVICE_HOST, SERVICE_PORT,
                          SERVICE_AUTH_KEY, logger=LOGGER.getChild("interface"))
AIINTERFACE.start()
