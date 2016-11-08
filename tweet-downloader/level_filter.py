import logging

class LevelFilter(logging.Filter):
    def __init__(self, exclusive_maximum, name=""):
        super(LevelFilter, self).__init__(name)
        self.max_level = exclusive_maximum

    def filter(self, record):
        #non-zero return means we log this message
        return 1 if record.levelno == self.max_level else 0