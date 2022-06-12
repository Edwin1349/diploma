from threading import Thread
import cv2
from queue import Queue

class FileVideoStream:
    def __init__(self, path, queueSize=600):
        self.thread = None
        self.stream_in = cv2.VideoCapture(path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.stream_out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)

    def start(self) -> object:
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                (grabbed, frame) = self.stream_in.read()
                if not grabbed:
                    self.stop()
                    return
                self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True

    def write(self, frame):
        self.stream_out.write(frame)