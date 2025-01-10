"""
ACTIONS = [
      "RAISE LEFT EYEBROW",
      "RAISE RIGHT EYEBROW",

      "CLOSE LEFT EYE",
      "CLOSE RIGHT EYE",

      "ROUND",
      "KISS",
      "SMILE WITH TEETH",
      "SMILE NO TEETH",
      "SAD",
      "SMIRK LEFT",
      "SMIRK RIGHT" 
]
"""

import cv2
import time
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import torch
import numpy as np
import random

def sort_boxes_by_area(boxes):
      def area(box):
            xmin, ymin, xmax, ymax = box.left(), box.top(), box.right(), box.bottom()
            return (xmax - xmin) * (ymax - ymin)
      
      sorted_boxes = sorted(boxes, key=area, reverse=True)
      return sorted_boxes

def draw(eyebrows, nose, eye, mouth, frame):
      #eye brows
      for i, pt in enumerate(eyebrows):
            #  right    left
            if i==2 or i==7:
                  cv2.circle(frame, pt, 1, (255, 0, 0), 3)
            # else:
            #       cv2.circle(frame, pt, 1, (0, 255, 255), 3)
      cv2.line(frame, eyebrows[2], eyebrows[7], (255, 255, 255), 1)

      #nose
      cv2.circle(image, nose, 4, (0, 255, 255), 2)

      #eye
      for i, pt in enumerate(eye):
            #  right    left  (inner)
            if i==3 or i==6:
                  cv2.circle(frame, pt, 1, (255, 0, 0), 3)
            #    topright  botright  topleft botleft
            elif i==1 or  i==5 or  i==8 or  i==10:
                  cv2.circle(frame, pt, 1, (0, 255, 255), 3)
      cv2.line(frame, eye[3], eye[6], (255, 255, 255), 1)
      cv2.line(frame, eye[1], eye[5], (255, 255, 255), 1)
      cv2.line(frame, eye[8], eye[10], (255, 255, 255), 1)

      #mouth
      cv2.line(frame, (mouth[11]), (mouth[2]), (0, 255, 255), 1)
      cv2.line(frame, (mouth[2]), (mouth[15]), (0, 255, 255), 1)
      cv2.line(frame, (mouth[11]), (mouth[13]), (0, 255, 255), 1)
      cv2.line(frame, (mouth[13]), (mouth[15]), (0, 255, 255), 1)
      cv2.line(frame, (mouth[11]), (mouth[17]), (0, 255, 255), 1)
      cv2.line(frame, (mouth[17]), (mouth[15]), (0, 255, 255), 1)
      cv2.line(frame, (mouth[11]), (mouth[8]), (0, 255, 255), 1)
      cv2.line(frame, (mouth[8]), (mouth[15]), (0, 255, 255), 1)
      for i, pt in enumerate(mouth):
            #  top1    bot2    right     top2     left    bot1
            if i==2 or i==8 or i==11 or i==13 or i==15 or i==17:
                  cv2.circle(frame, pt, 1, (255, 0, 0), 2)
            #    outlier
            # elif i != 5:
            #       cv2.circle(frame, pt, 1, (0, 255, 255), 2)

      return frame

def dist_point2line(C, A, B):
      """
      Dist point C -> line AB
      """
      x1, y1 = A
      x2, y2 = B
      x0, y0 = C

      numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
      denominator = torch.sqrt((y2 - y1)**2 + (x2 - x1)**2)
      distance = numerator / denominator

      return distance

def dist_point2point(A, B):
      x1, y1 = A
      x2, y2 = B

      distance = torch.sqrt((y2 - y1)**2 + (x2 - x1)**2)

      return distance

class MouthModel(torch.nn.Module):
      def __init__(self):
            super().__init__()
      def forward(self, landmarks):
            nose, right_eyebrow, left_eyebrow, topright_eye, bottomright_eye, topleft_eye, bottomleft_eye,  right_innereye, left_innereye, top_lip1, top_lip2, bottom_lip1, bottom_lip2, right_mouth, left_mouth = landmarks[0]
            
            # eyebrows
            yright_eyebrow = dist_point2line(right_eyebrow, right_innereye, left_innereye)
            yleft_eyebrow = dist_point2line(left_eyebrow, right_innereye, left_innereye)
            rightleft_eb_ratio = yright_eyebrow / yleft_eyebrow

            # eye
            dist_righteyes = dist_point2point(topright_eye, bottomright_eye)
            dist_lefteyes = dist_point2point(topleft_eye, bottomleft_eye)
            dist_toprighteye_righteb = dist_point2point(topright_eye, right_eyebrow)
            dist_toplefteye_lefteb = dist_point2point(topleft_eye, left_eyebrow)
            righteye_eb_ratio = dist_righteyes / dist_toprighteye_righteb
            lefteye_eb_ratio = dist_lefteyes / dist_toplefteye_lefteb

            # mouth
            h1 = torch.sqrt(torch.sum((bottom_lip2 - top_lip1) ** 2))
            h2 = torch.sqrt(torch.sum((bottom_lip1 - top_lip2) ** 2))
            w = torch.sqrt(torch.sum((left_mouth - right_mouth) ** 2))

            h1w_ratio = h1 / w
            h2h1_ratio = h2 / h1
            
            yright_mouth = dist_point2line(right_mouth, right_innereye, left_innereye)
            yleft_mouth = dist_point2line(left_mouth, right_innereye, left_innereye)
            ybot1 = dist_point2line(bottom_lip1, right_innereye, left_innereye)
            ytop1 = dist_point2line(top_lip1, right_innereye, left_innereye)
            ytop2 = dist_point2line(top_lip2, right_innereye, left_innereye)
            ynose = dist_point2line(nose, right_innereye, left_innereye)

            yright_bot1 = yright_mouth - ybot1
            yleft_bot1 = yleft_mouth - ybot1
            yright_top1 = yright_mouth - ytop1
            yleft_top1 = yleft_mouth - ytop1
            yright_top2 = yright_mouth - ytop2
            yleft_top2 = yleft_mouth - ytop2
            nosetop2 = ynose / ytop2
            

            res = torch.cat((rightleft_eb_ratio.unsqueeze(0), righteye_eb_ratio.unsqueeze(0), lefteye_eb_ratio.unsqueeze(0), h1.unsqueeze(0), h1w_ratio.unsqueeze(0), h2h1_ratio.unsqueeze(0), nosetop2.unsqueeze(0), yright_bot1.unsqueeze(0), yleft_bot1.unsqueeze(0), yright_top1.unsqueeze(0), yleft_top1.unsqueeze(0), yright_top2.unsqueeze(0), yleft_top2.unsqueeze(0)), dim=0)
            
            return res


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# grab the indexes of the facial landmarks for the mouth
mymodel = MouthModel()



list_actions = [
      ["RAISE RIGHT EYEBROW", "RAISE LEFT EYEBROW", "CLOSE RIGHT EYE", "CLOSE LEFT EYE", "RAISE LEFT EYEBROW + SAD", "CLOSE RIGHT EYE + SMILE NO TEETH"], 
      ["SMIRK LEFT", "KISS", "RAISE RIGHT EYEBROW", "SAD", "ROUND + CLOSE RIGHT EYE", "KISS + CLOSE LEFT EYE", "RAISE LEFT EYEBROW", "SMILE WITH TEETH"], 
      ["RAISE LEFT EYEBROW + ROUND", "SMILE WITH TEETH", "KISS", "SAD", "SMILE NO TEETH", "KISS", "SAD", "SMIRK RIGHT"]
      ]

actions = random.choice(list_actions)

TIME_LIMIT = len(actions) * 8  # time per expression


current_i = 0
alert = ""
pass_time = 0
display_text = False
quit = False
start_time = time.time()
print(">>>>>>>> START")
cap = cv2.VideoCapture(0)
while time.time() - start_time < TIME_LIMIT:
      ret, frame = cap.read()
      if not ret:
            break

      # Process frame here
      image = imutils.resize(frame, width=1000)
      w, h = image.shape[:2]
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      cv2.putText(image, "Challenge: ", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
      cv2.putText(image, list_actions[current_i], (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

      cv2.putText(image, alert, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

      cur_time_text = "Time: " + str(round(time.time() - start_time, 2)) + "/" + str(TIME_LIMIT) + "s"
      cv2.putText(image, cur_time_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

      cv2.putText(image, "Current status:", (30, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

      try:
            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # sorted rect by area:
            sorted_rects = sort_boxes_by_area(rects)

            rect = sorted_rects[0]

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eyebrowns = shape[17:27]
            nose = shape[30]
            eye = shape[36:50]
            mouth = shape[49:68]

            # visualize
            image = draw(eyebrowns, nose, eye, mouth, image)

            # input: nose, right_eyebrow, left_eyebrow, topright_eye, bottomright_eye, topleft_eye, bottomleft_eye,  right_innereye, left_innereye, top_lip1, top_lip2, bottom_lip1, bottom_lip2, right_mouth, left_mouth
            input = np.array([nose, 
                              eyebrowns[2], eyebrowns[7], 
                              eye[1], eye[5], eye[8], eye[10], eye[3], eye[6], 
                              mouth[2], mouth[13], mouth[17], mouth[8], mouth[11], mouth[15]])
            input = (torch.tensor(input)).unsqueeze(0)

            # preds
            scores = mymodel(input)

            rightleft_eb_ratio, righteye_eb_ratio, lefteye_eb_ratio, h1, h1w, h2h1, nosetop2, yright_bot1, yleft_bot1, yright_top1, yleft_top1, yright_top2, yleft_top2 = scores

            ### EYEBROWS
            # right raised
            if rightleft_eb_ratio > 1.1:
                  pred_eyebrow = "RAISE RIGHT EYEBROW"
            # left raised
            elif rightleft_eb_ratio < 0.9:
                  pred_eyebrow = "RAISE LEFT EYEBROW"
            else:
                  pred_eyebrow = "NONE"
            cv2.putText(image, "- Eyebrow raised: " + pred_eyebrow, (30, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            ### EYES
            # right closed
            if righteye_eb_ratio < 0.15 and lefteye_eb_ratio > 0.15:
                  pred_eye = "CLOSE RIGHT EYE"
            # left closed
            elif lefteye_eb_ratio < 0.15 and righteye_eb_ratio > 0.15:
                  pred_eye = "CLOSE LEFT EYE"
            elif righteye_eb_ratio < 0.15 and lefteye_eb_ratio < 0.15:
                  pred_eye = "BOTH"
            else:
                  pred_eye = "None"
            cv2.putText(image, "- Eye closed: " + pred_eye, (30, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            ### MOUTH
            # nhech mep
            if h2h1 < 0.2 and (yright_top2 < -0.2*h1 and yleft_top2 > 0.1*h1):
                  pred_mouth = "SMIRK RIGHT"
            elif h2h1 < 0.2 and (yright_top2 > 0.1*h1 and yleft_top2 < -0.2*h1):
                  pred_mouth = "SMIRK LEFT"
            # round
            elif h1w > 1.5 and h2h1 > 0.5:
                  pred_mouth = "ROUND"
            # sad
            elif h2h1 < 0.3 and yright_top2 > 0.15*h1 and yleft_top2 > 0.15*h1 and yright_bot1 > 0.05*h1 and yleft_bot1 > 0.05*h1 and nosetop2 > 0.3:
                  pred_mouth = "SAD"
            # smile
            elif h2h1 > 0.3 and h1w < 0.55 and torch.abs(yright_top2) < 0.1*h1  and torch.abs(yleft_top2) < 0.1*h1:
                  pred_mouth = "SMILE WITH TEETH" 
            elif h2h1 < 0.3 and h1w < 0.55 and yright_top2 < -0.1*h1  and yleft_top2 < -0.1*h1:
                  pred_mouth = "SMILE NO TEETH"
            # kiss
            elif h1w > 0.9 and h2h1 < 0.25:
                  pred_mouth = "KISS"
            # normal
            else:
                  pred_mouth = "NORMAL"
            cv2.putText(image, "- Mouth: " + pred_mouth, (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            preds = [pred_eyebrow, pred_eye, pred_mouth]

            expressions = [list_actions[current_i]]
            if len(list_actions[current_i].split("+")) > 1:
                  expressions = [exp.strip() for exp in list_actions[current_i].split("+")]

            if all(exp in preds for exp in expressions):
                  current_i += 1
                  display_text = True
                  alert = "PASSED"
                  pass_time = time.time()

            if current_i == len(list_actions):
                  break

            cv2.imshow('Webcam', image)
            if cv2.waitKey(1) & 0xFF == 27:
                  quit = True
                  break
      except:
            cv2.putText(image, "NO FACE", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            cv2.imshow('Webcam', image)

            if cv2.waitKey(1) & 0xFF == 27:
                  quit = True
                  break

      if display_text:
            current_time = time.time()
            if current_time - pass_time < 1:
                  alert = "PASSED"
            else:
                  alert = ""
                  display_text = False

if not quit and current_i < len(list_actions):
      text_time = f"Total time: {TIME_LIMIT}s"
      while True:
            final_frame = np.ones((w, h, 3), dtype=np.uint8) * 255

            cv2.putText(final_frame, "TIME'S UP, FAILED!", (190, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(final_frame, text_time, (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('Webcam', final_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                  break

if current_i >= len(list_actions):
      tol_time = time.time() - start_time
      text_time = f"Total time: {round(tol_time, 2)}s"
      while True:
            final_frame = np.ones((w, h, 3), dtype=np.uint8) * 255

            cv2.putText(final_frame, "FINISHED!", (310, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(final_frame, text_time, (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('Webcam', final_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                  break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
