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
import math
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import argparse

# Tạo đối tượng ArgumentParser
parser = argparse.ArgumentParser(description='Thay đổi giá trị biến TYPE.')
parser.add_argument('--type', type=str, default="ape")

parser.add_argument('--num_exps', type=int, default=12)

# Phân tích các đối số
args = parser.parse_args()
# Thay đổi giá trị của biến TYPE
TYPE = args.type
num_expressions = args.num_exps


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

def paste_image(l_img, s_img, offset=(0, 0), h_ratio=5):
      x_offset, y_offset = offset

      if h_ratio <= 0:
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
      else:
            new_height = l_img.shape[0] // h_ratio  
            aspect_ratio = s_img.shape[1] / s_img.shape[0]  
            new_width = new_height * aspect_ratio
            s_img = cv2.resize(s_img, (int(new_width), int(new_height)))
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

      return l_img

def add_transparent_image(background, foreground, x_offset=0, y_offset=0):
      bg_h, bg_w, bg_channels = background.shape
      fg_h, fg_w, fg_channels = foreground.shape

      assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
      assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

      # center by default
      if x_offset is None: x_offset = (bg_w - fg_w) // 2
      if y_offset is None: y_offset = (bg_h - fg_h) // 2

      w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
      h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

      if w < 1 or h < 1: return

      # clip foreground and background images to the overlapping regions
      bg_x = max(0, x_offset)
      bg_y = max(0, y_offset)
      fg_x = max(0, x_offset * -1)
      fg_y = max(0, y_offset * -1)
      foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
      background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

      # separate alpha and color channels from the foreground image
      foreground_colors = foreground[:, :, :3]
      alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

      # construct an alpha_mask that matches the image shape
      alpha_mask = alpha_channel[:, :, np.newaxis]

      # combine the background with the overlay image weighted by alpha
      composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

      # overwrite the section of the background image that has been updated
      background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

      return background

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
      dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
      pts= []
      for i in  np.arange(0,dist,gap):
            r=i/dist
            x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
            y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
            p = (x,y)
            pts.append(p)

      if style=='dotted':
            for p in pts:
                  cv2.circle(img,p,thickness,color,-1)
      else:
            s=pts[0]
            e=pts[0]
            i=0
            for p in pts:
                  s=e
                  e=p
                  if i%2==1:
                        cv2.line(img,s,e,color,thickness)
                  i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
      s=pts[0]
      e=pts[0]
      pts.append(pts.pop(0))
      for p in pts:
            s=e
            e=p
            drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
      pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
      drawpoly(img,pts,color,thickness,style)

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





print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mymodel = MouthModel()

ACTIONS = []
for imgname in os.listdir("illustrations/ape"):
      action = imgname.split('.')[0]
      ACTIONS.append(action)

# n_actions = 12
tmp_actions = ACTIONS
random.shuffle(tmp_actions)
list_actions = random.sample(tmp_actions, num_expressions)

TIME_LIMIT = len(list_actions) * 50  # time per expression
TIME_COUNT_DOWN_START = 3

current_i = 0
alert = ""
pass_time = 0
display_text = False
quit = False

countdown_started = False









def start_countdown():
    global start_time_0
    start_time_0 = time.time()
    global countdown_started
    countdown_started = True
    start_button.config(state=tk.DISABLED)  # Disable the button after starting
    start_button.pack_forget()
    title_label.pack_forget()
    title_label2.pack_forget()
    frame.pack_forget()

    update_frame()  # Start updating the webcam feed

def update_label(image):
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

def process_game_frames(start_time):
      global current_i, alert, display_text, pass_time
      bg = cv2.imread("simple-white-background-a8m5kg369yz3b9xh.png", cv2.IMREAD_UNCHANGED)
      ret, frame = cap.read()

      # Process frame here
      image = imutils.resize(frame, width=1100)

      w, h = image.shape[:2]
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      image = add_transparent_image(image, bg, 0, 0)

      cur_time_text = "Time: " + str(round(time.time() - start_time, 2)) + "/" + str(TIME_LIMIT) + "s"
      cv2.putText(image, cur_time_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

      cv2.putText(image, "Challenge: ", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

      if len(list_actions[current_i].split("+")) > 1:
            first_action = list_actions[current_i].split("+")[0].strip()
            second_action = list_actions[current_i].split("+")[1].strip()
            cv2.putText(image, first_action + ",", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(image, second_action, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

      else:
            cv2.putText(image, list_actions[current_i], (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

      cv2.putText(image, "PUT YOUR FACE HERE!", (500, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

      if TYPE == "ape":
            illustration = cv2.imread("illustrations/ape/" + list_actions[current_i] + ".jpg")
            image = paste_image(image, illustration, (30, 220), 1.5)
      else:
            imgpath = "illustrations/emoji/" + list_actions[current_i] + ".png"
            illustration = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
            illustration = cv2.resize(illustration, (300, 300))
            image = add_transparent_image(image, illustration, 30, 220)

      cv2.putText(image, alert, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4, 1)

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

            input = np.array([nose, 
                        eyebrowns[2], eyebrowns[7], 
                        eye[1], eye[5], eye[8], eye[10], eye[3], eye[6], 
                        mouth[2], mouth[13], mouth[17], mouth[8], mouth[11], mouth[15]])
            input = (torch.tensor(input)).unsqueeze(0)

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

            ### EYES
            # right closed
            if righteye_eb_ratio < 0.15 and lefteye_eb_ratio > 0.15:
                  pred_eye = "BLINK RIGHT EYE"
            # left closed
            elif lefteye_eb_ratio < 0.15 and righteye_eb_ratio > 0.15:
                  pred_eye = "BLINK LEFT EYE"
            elif righteye_eb_ratio < 0.15 and lefteye_eb_ratio < 0.15:
                  pred_eye = "BOTH"
            else:
                  pred_eye = "None"

            ### MOUTH
            # nhech mep
            if h2h1 < 0.2 and (yright_top2 < -0.2*h1 and yleft_top2 > 0.1*h1):
                  pred_mouth = "SMIRK RIGHT"
            elif h2h1 < 0.2 and (yright_top2 > 0.1*h1 and yleft_top2 < -0.2*h1):
                  pred_mouth = "SMIRK LEFT"
            # round
            elif h1w > 1.5 and h2h1 > 0.5:
                  pred_mouth = "MOUTH ROUND"
            # sad
            elif h2h1 < 0.3 and yright_top2 > 0.15*h1 and yleft_top2 > 0.15*h1 and yright_bot1 > 0.05*h1 and yleft_bot1 > 0.05*h1 and nosetop2 > 0.3:
                  pred_mouth = "MOUTH SAD"
            # smile
            elif h2h1 > 0.3 and h1w < 0.55 and torch.abs(yright_top2) < 0.1*h1  and torch.abs(yleft_top2) < 0.1*h1:
                  pred_mouth = "SMILE WITH TEETH" 
            elif h2h1 < 0.3 and h1w < 0.55 and yright_top2 < -0.1*h1  and yleft_top2 < -0.1*h1:
                  pred_mouth = "SMILE NO TEETH"
            # kiss
            elif h1w > 0.9 and h2h1 < 0.25:
                  pred_mouth = "MOUTH KISS"
            # normal
            else:
                  pred_mouth = "NORMAL"

            preds = [pred_eyebrow, pred_eye, pred_mouth]

            expressions = [list_actions[current_i]]
            if len(list_actions[current_i].split("+")) > 1:
                  expressions = [exp.strip() for exp in list_actions[current_i].split("+")]

            if all(exp in preds for exp in expressions):
                  current_i += 1
                  display_text = True
                  alert = "PASSED"
                  pass_time = time.time()

      except:
            cv2.putText(image, "NO FACE", (650, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

      if display_text:
            current_time = time.time()
            if current_time - pass_time < 1:
                  alert = "PASSED"
            else:
                  alert = ""
                  display_text = False

      # Update the label with the processed image
      update_label(image)

      if time.time() - start_time > TIME_LIMIT and current_i < len(list_actions):
            text_time = f"Total time: {TIME_LIMIT}s"
            # while True:
            final_frame = np.ones((w, h, 3), dtype=np.uint8) * 255

            cv2.putText(final_frame, "TIME'S UP, FAILED!", (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(final_frame, text_time, (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

            update_label(final_frame)

      if current_i >= len(list_actions):
            tol_time = time.time() - start_time
            text_time = f"Total time: {round(tol_time, 2)}s"
            # while True:
            final_frame = np.ones((w, h, 3), dtype=np.uint8) * 255

            cv2.putText(final_frame, "FINISHED!", (360, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(final_frame, text_time, (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

            update_label(final_frame)
            
      video_label.after(10, update_frame)

def update_frame():
    if countdown_started:
        ret, frame = cap.read()
        if ret:
            if time.time() - start_time_0 < TIME_COUNT_DOWN_START:
                # Countdown logic
                image = imutils.resize(frame, width=1100)
                ksize = (15, 15)
                image = cv2.blur(image, ksize)

                current_time = time.time() - start_time_0
                remaining_time = max(0, math.ceil(TIME_COUNT_DOWN_START - current_time))

                cv2.putText(image, "GAME STARTS IN:", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
                cv2.putText(image, str(remaining_time), (500, 480), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)

                # Update the label with the countdown image
                update_label(image)

                if remaining_time > 0:
                    video_label.after(10, update_frame)  # Continue updating frame
                else:
                    # Start the main game logic after countdown
                    process_game_frames(start_time_0 + TIME_COUNT_DOWN_START)

            else:
                process_game_frames(start_time_0 + TIME_COUNT_DOWN_START)


# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a simple GUI with tkinter
root = tk.Tk()
root.title("Expression Game")

start_time_0 = time.time()

frame = tk.Frame(root)
frame.pack(expand=True)  # Center the frame

# Title label
title_label = tk.Label(frame, text="Expression Game", font=("Helvetica", 48))
title_label.pack(pady=10)  # Reduced padding

# Subtitle label
title_label2 = tk.Label(frame, text="Try to pass all the challenges!", font=("Helvetica", 24))
title_label2.pack(pady=5)  # Reduced padding

# Start button
start_button = tk.Button(frame, text="Start Game", command=start_countdown, height=5, width=20)
start_button.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

root.geometry("1200x800")
root.mainloop()

cap.release()
cv2.destroyAllWindows()