from classification_model import *
from detection_model import *
from FileVideoStream import *
from imutils.video import FPS
import time
import sys

def load_classificator(path):
    model = TrafficSignNet().to(device)
    model = torch.load(path)
    model.eval()

    try:
        image_BGR = cv2.imread('C:/Users/Orest/Documents/BCW/code/python/jojo1.png')
        image_BGR = val_transform(image_BGR)
        outputs = model(image_BGR[None, ...].to(device))
    except:
        pass

    return model


def draw(frame, boxes):
    colour_box_current = [0, 251, 61]

    for box in boxes:
        cv2.rectangle(frame, (box.x_min, box.y_min),
                      (box.x_min + box.box_width, box.y_min + box.box_height),
                      colour_box_current, 2)

        text_box_current = '{}: {:.4f}'.format(box.ts_class,
                                               box.confidence)

        cv2.putText(frame, text_box_current, (box.x_min, box.y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current,
                    2)
        #print('class: ', box.ts_class, 'confidence: ', box.confidence)

    cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Object', frame)
    #cv2.imwrite('result.jpg', frame)


def run(path, classifier, detector):
    with open('classes.names') as f:
        labels = [line.strip() for line in f]

    fvs = FileVideoStream(path).start()
    time.sleep(1.0)

    fps = FPS().start()

    while fvs.more():
        frame = fvs.read()

        results = detector.detect(frame)
        boxes = list()

        if len(results) > 0:
            for i in results.flatten():
                try:
                    if frame is not None:
                        box = Box(detector.bounding_boxes[i][0], detector.bounding_boxes[i][1],
                                  detector.bounding_boxes[i][2], detector.bounding_boxes[i][3],
                                  detector.class_numbers[i])

                        c_ts = frame[box.y_min: box.y_min + int(box.box_height), box.x_min: box.x_min + int(box.box_width), :]
                        c_ts = val_transform(c_ts)

                        outputs = classifier(c_ts[None, ...].to(device))
                        _, predicted = torch.max(outputs.data, 1)
                        prob = torch.nn.functional.softmax(outputs, dim=1)
                        if prob[0][types[detector.class_numbers[i]]].max() > 0.8:
                            box.set_class(classes[types[detector.class_numbers[i]][prob[0][types[detector.class_numbers[i]]].argmax()]],
                                          prob[0][types[detector.class_numbers[i]]].max())
                        boxes.append(box)
                except:
                    continue

        draw(frame, boxes)
        #fvs.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

        fps.update()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    fvs.stop()


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    classifier = load_classificator("tsnet_epochs30.pth")
    detector = YOLO("C:/darknet/build/darknet/x64/cfg/yolov3_ts_test.cfg",
                    "C:/darknet/build/darknet/x64/backup/yolov3_ts_train_8000.weights")

    if len(sys.argv) > 1:
        print(sys.argv[1])
        run(sys.argv[1], classifier, detector)
    else:
        print(sys.argv[0])
        run("C:\\Users\\Orest\\Documents\\BCW\\test_video.mp4", classifier, detector)

