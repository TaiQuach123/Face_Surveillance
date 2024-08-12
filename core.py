from detector import RetinaFaceDetector
from recognizer import ArcFaceRecognizer
from Tracking.sort import Sort
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

cudnn.benchmark = True
torch.set_grad_enabled(False)

class FaceSurveillanceCore:
    def __init__(self, face_detector_name = "mobilenet", face_recognizer_name = "iresnet100", tracker_name = "sort"):
        self.face_detector = RetinaFaceDetector(name=face_detector_name, pretrained_path = "weights/detection/mobilenet_100.pth")
        self.face_recognizer = ArcFaceRecognizer(name=face_recognizer_name, face_db_file='db.pkl')
        self.mot_tracker = Sort(max_age=15)
        self.trackid_to_name = {}
        
    def process_single_video(self, source):
        cap = cv2.VideoCapture(source)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter('test_camera.avi', cv2.VideoWriter_fourcc(*"MJPG"), 20, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if frame is None:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frame, dets = self.face_detector.detect_single_image(frame)
            faces_bboxes = dets[:, :5]
            
            faces, faces_landms = self.face_detector.extract_faces_landms(raw_frame, dets)
            aligned_faces = []
            right_eyes = faces_landms[:, :2]
            left_eyes = faces_landms[:, 2:4]

            for i in range(len(faces)):
                aligned_face, _ = self.face_detector.align_face(faces[i], left_eyes[i], right_eyes[i])
                aligned_faces.append(aligned_face)
            
            identity_names = []
            scores = []
            
            for i, face in enumerate(aligned_faces):

                name, cosine_score = self.face_recognizer.find_person(face)
                identity_names.append(name)
                scores.append(cosine_score)
            

            faces_bboxes = np.array(faces_bboxes) if len(faces_bboxes) != 0 else np.empty((0,5))
            trackers = self.mot_tracker.update(faces_bboxes)
            
            identity_names = reversed(identity_names)
            scores = reversed(scores)
            for name, tracker, score in zip(identity_names, trackers, scores):
                tracker = [int(i) for i in tracker]
                x1, y1, x2, y2, trackid = tracker
                #print(name, ' ---' , score, ' --- ', tracker)

                cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0,0,255), 2)

                if not self.trackid_to_name.get(trackid) and name != "unknown":
                    self.trackid_to_name[trackid] = name
                
                if name == "unknown" and self.trackid_to_name.get(trackid):
                    name = self.trackid_to_name[trackid]

                text = f"{name}_{trackid}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                text_w, text_h = text_size
                cv2.rectangle(raw_frame, (x1, y1), (x1+text_w, y1 - text_h), (0,0,255), -1)
                cv2.putText(raw_frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('frame', cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) == ord('q'):
                break
            out.write(cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_surveillance = FaceSurveillanceCore()
    face_surveillance.process_single_video(0)
            