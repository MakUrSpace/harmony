import fcntl
import os
import sys

LOCK_FILE = "/tmp/harmony_observer.lock"

class FileLock:
    def __init__(self):
        self.lock_file = open(LOCK_FILE, "w")

    def acquire(self):
        try:
            fcntl.lockf(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            print(f"Another instance is already running. Exiting.")
            sys.exit(1)

    def release(self):
        try:
            fcntl.lockf(self.lock_file, fcntl.LOCK_UN)
        except IOError:
            pass
