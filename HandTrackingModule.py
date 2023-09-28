import cv2
import mediapipe
import time

class HandDector():
    def __init__(self, mode=False, max_hands=2,model_complexity=1,detection_conf = 0.5, track_confi = 0.5):

        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_conf = detection_conf
        self.track_conf = track_confi

        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands( static_image_mode=self.mode,
                                        max_num_hands=self.max_hands,
                                        model_complexity=self.model_complexity,
                                        min_detection_confidence=self.detection_conf,
                                        min_tracking_confidence=self.track_conf)
        self.myDraw = mediapipe.solutions.drawing_utils

    def FindHands(self, img, draw = True):

        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_new)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.myDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img


# for id, lm in enumerate(handLms.landmark):
                #     h, w, c = img.shape
                #     cx, cy = int(lm.x * w), int(lm.y * h)
               #     #print(id, cx, cy)
def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDector()
    while True:
        _, img = cap.read()
        img = detector.FindHands(img, draw=True)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv2.imshow('Result', img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()