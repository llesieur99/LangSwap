import os
import json
from typing import List
import logging
import hashlib
import tempfile

class Logger:
    def __init__(self, directory) -> None:
        self.directory = directory

        self.file_logger = self.setup_logger()

    def setup_logger(self, log_file='logs.log'):
        key = hashlib.sha256(os.path.realpath(self.directory).encode()).hexdigest()[:16]
        logger = logging.getLogger(f'langswap.job.{key}')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_file = os.path.join(self.directory, log_file)
        if any(isinstance(handler, logging.FileHandler)
               and handler.baseFilename == os.path.abspath(log_file)
               for handler in logger.handlers):
            return logger
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    def log_json(self, file_name, data: List):
        log_text = os.path.realpath(os.path.join(self.directory, file_name))
        if os.path.commonpath((os.path.realpath(self.directory), log_text)) != os.path.realpath(self.directory):
            raise ValueError("JSON logs must stay within the job directory.")
        fd, temporary = tempfile.mkstemp(dir=self.directory, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            os.replace(temporary, log_text)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
