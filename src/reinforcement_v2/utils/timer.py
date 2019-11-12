import time


class Timer():
    def __init__(self):
        self.elapsed_time = 0.0
        self.start_time = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.elapsed_time += time.time() - self.start_time
        self.start_time = 0.0