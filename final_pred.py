# Importing Libraries
import numpy as np
import math
import cv2

import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant

ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
import tkinter as tk
from PIL import Image, ImageTk

offset = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application :

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image_with_landmarks = None
        self.model = load_model('cnn8grps_rad1_model.h5')
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = []
        for i in range(10):
            self.ten_prev_char.append(" ")

        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")

        # Create main window with dark theme
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1400x750")

        # Set dark background color
        self.root.configure(bg='#2b2b2b')

        # Main title
        self.T = tk.Label(self.root, bg='#2b2b2b', fg='#ffffff')
        self.T.place(x=500, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        # Title for Normal Feed
        self.T_normal = tk.Label(self.root, bg='#2b2b2b', fg='#00ff00')
        self.T_normal.place(x=200, y=70)
        self.T_normal.config(text="Normal Camera Feed", font=("Courier", 16, "bold"))

        # Title for Landmark Feed
        self.T_landmark = tk.Label(self.root, bg='#2b2b2b', fg='#00ff00')
        self.T_landmark.place(x=900, y=70)
        self.T_landmark.config(text="Feed with Hand Landmarks", font=("Courier", 16, "bold"))

        # Video panel for normal feed
        self.panel_normal = tk.Label(self.root, bg='#2b2b2b')
        self.panel_normal.place(x=50, y=100, width=480, height=640)

        # Video panel for feed with landmarks
        self.panel_landmark = tk.Label(self.root, bg='#2b2b2b')
        self.panel_landmark.place(x=600, y=100, width=480, height=640)

        # Current Symbol panel
        self.panel3 = tk.Label(self.root, bg='#2b2b2b', fg='#00ff00')  # Green text for character
        self.panel3.place(x=1120, y=300)

        # Character label
        self.T1 = tk.Label(self.root, bg='#2b2b2b', fg='#ffffff')
        self.T1.place(x=1120, y=250)
        self.T1.config(text="Character :", font=("Courier", 20, "bold"))

        # Sentence panel
        self.panel5 = tk.Label(self.root, bg='#2b2b2b', fg='#00ff00')  # Green text for sentence
        self.panel5.place(x=1120, y=400)

        # Sentence label
        self.T3 = tk.Label(self.root, bg='#2b2b2b', fg='#ffffff')
        self.T3.place(x=1120, y=350)
        self.T3.config(text="Sentence :", font=("Courier", 20, "bold"))

        # Speak button
        self.speak = tk.Button(self.root, bg='#404040', fg='#ffffff', activebackground='#505050',
                               activeforeground='#ffffff', relief='flat')
        self.speak.place(x=1120, y=500)
        self.speak.config(text="Speak", font=("Courier", 20), wraplength=100, command=self.speak_fun)

        # Clear button
        self.clear = tk.Button(self.root, bg='#404040', fg='#ffffff', activebackground='#505050',
                               activeforeground='#ffffff', relief='flat')
        self.clear.place(x=1120, y=570)
        self.clear.config(text="Clear", font=("Courier", 20), wraplength=100, command=self.clear_fun)

        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"

        self.video_loop()

    def draw_landmarks_on_hand(self, frame, pts, bbox):
        """Draw hand landmarks and connections directly on the frame"""
        x, y, w, h = bbox
        
        # Create a copy of the frame to draw on
        frame_with_landmarks = frame.copy()
        
        # Draw connections between landmarks
        # Thumb (0-4)
        for t in range(0, 4):
            if t+1 < len(pts):
                cv2.line(frame_with_landmarks, 
                        (int(pts[t][0] + x - offset), int(pts[t][1] + y - offset)),
                        (int(pts[t+1][0] + x - offset), int(pts[t+1][1] + y - offset)),
                        (0, 255, 0), 3)
        
        # Index finger (5-8)
        for t in range(5, 8):
            if t+1 < len(pts):
                cv2.line(frame_with_landmarks,
                        (int(pts[t][0] + x - offset), int(pts[t][1] + y - offset)),
                        (int(pts[t+1][0] + x - offset), int(pts[t+1][1] + y - offset)),
                        (0, 255, 0), 3)
        
        # Middle finger (9-12)
        for t in range(9, 12):
            if t+1 < len(pts):
                cv2.line(frame_with_landmarks,
                        (int(pts[t][0] + x - offset), int(pts[t][1] + y - offset)),
                        (int(pts[t+1][0] + x - offset), int(pts[t+1][1] + y - offset)),
                        (0, 255, 0), 3)
        
        # Ring finger (13-16)
        for t in range(13, 16):
            if t+1 < len(pts):
                cv2.line(frame_with_landmarks,
                        (int(pts[t][0] + x - offset), int(pts[t][1] + y - offset)),
                        (int(pts[t+1][0] + x - offset), int(pts[t+1][1] + y - offset)),
                        (0, 255, 0), 3)
        
        # Pinky finger (17-20)
        for t in range(17, 20):
            if t+1 < len(pts):
                cv2.line(frame_with_landmarks,
                        (int(pts[t][0] + x - offset), int(pts[t][1] + y - offset)),
                        (int(pts[t+1][0] + x - offset), int(pts[t+1][1] + y - offset)),
                        (0, 255, 0), 3)
        
        # Palm connections
        if len(pts) > 17:
            cv2.line(frame_with_landmarks,
                    (int(pts[5][0] + x - offset), int(pts[5][1] + y - offset)),
                    (int(pts[9][0] + x - offset), int(pts[9][1] + y - offset)),
                    (0, 255, 0), 3)
            cv2.line(frame_with_landmarks,
                    (int(pts[9][0] + x - offset), int(pts[9][1] + y - offset)),
                    (int(pts[13][0] + x - offset), int(pts[13][1] + y - offset)),
                    (0, 255, 0), 3)
            cv2.line(frame_with_landmarks,
                    (int(pts[13][0] + x - offset), int(pts[13][1] + y - offset)),
                    (int(pts[17][0] + x - offset), int(pts[17][1] + y - offset)),
                    (0, 255, 0), 3)
            cv2.line(frame_with_landmarks,
                    (int(pts[0][0] + x - offset), int(pts[0][1] + y - offset)),
                    (int(pts[5][0] + x - offset), int(pts[5][1] + y - offset)),
                    (0, 255, 0), 3)
            cv2.line(frame_with_landmarks,
                    (int(pts[0][0] + x - offset), int(pts[0][1] + y - offset)),
                    (int(pts[17][0] + x - offset), int(pts[17][1] + y - offset)),
                    (0, 255, 0), 3)
        
        # Draw landmarks as circles
        for i in range(len(pts)):
            cv2.circle(frame_with_landmarks, 
                      (int(pts[i][0] + x - offset), int(pts[i][1] + y - offset)), 
                      4, (0, 0, 255), -1)  # Filled red circles
        
        # Draw bounding box around hand
        cv2.rectangle(frame_with_landmarks, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 2)
        
        return frame_with_landmarks

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                self.root.after(10, self.video_loop)
                return
                
            cv2image = cv2.flip(frame, 1)
            
            # Create a copy for the landmark feed
            cv2image_with_landmarks = cv2image.copy()
            
            if cv2image.any():
                # Find hands in the frame - cvzone returns (hands, frame)
                hands, _ = hd.findHands(cv2image, draw=False, flipType=True)
                cv2image_copy = np.array(cv2image)
                
                if hands and len(hands) > 0:
                    hand = hands[0]  # Get the first hand
                    
                    # Get bbox from hand dictionary
                    if 'bbox' in hand:
                        x, y, w, h = hand['bbox']
                        
                        # Ensure coordinates are within frame boundaries
                        x = max(0, x)
                        y = max(0, y)
                        
                        # Extract the hand region for landmark detection
                        y_start = max(0, y - offset)
                        y_end = min(cv2image.shape[0], y + h + offset)
                        x_start = max(0, x - offset)
                        x_end = min(cv2image.shape[1], x + w + offset)
                        
                        image = cv2image_copy[y_start:y_end, x_start:x_end]
                        
                        if image.size > 0:
                            # Find hand landmarks in the extracted region
                            handz, _ = hd2.findHands(image, draw=False, flipType=True)
                            
                            if handz and len(handz) > 0:
                                hand_landmarks = handz[0]
                                
                                # Get landmarks list
                                if 'lmList' in hand_landmarks:
                                    self.pts = hand_landmarks['lmList']
                                    
                                    # Draw landmarks on the landmark feed copy
                                    cv2image_with_landmarks = self.draw_landmarks_on_hand(cv2image_with_landmarks, self.pts, (x, y, w, h))
                                    
                                    # Create a white image for model prediction
                                    white = np.ones((400, 400, 3), np.uint8) * 255
                                    
                                    # Calculate offset for white image
                                    os_x = ((400 - w) // 2) - 15
                                    os_y = ((400 - h) // 2) - 15
                                    
                                    # Draw landmarks on white background for model input
                                    if self.pts and len(self.pts) > 20:
                                        # Draw connections on white background
                                        for t in range(0, 4):
                                            if t+1 < len(self.pts):
                                                cv2.line(white, 
                                                        (int(self.pts[t][0] + os_x), int(self.pts[t][1] + os_y)),
                                                        (int(self.pts[t+1][0] + os_x), int(self.pts[t+1][1] + os_y)),
                                                        (0, 255, 0), 3)
                                        for t in range(5, 8):
                                            if t+1 < len(self.pts):
                                                cv2.line(white,
                                                        (int(self.pts[t][0] + os_x), int(self.pts[t][1] + os_y)),
                                                        (int(self.pts[t+1][0] + os_x), int(self.pts[t+1][1] + os_y)),
                                                        (0, 255, 0), 3)
                                        for t in range(9, 12):
                                            if t+1 < len(self.pts):
                                                cv2.line(white,
                                                        (int(self.pts[t][0] + os_x), int(self.pts[t][1] + os_y)),
                                                        (int(self.pts[t+1][0] + os_x), int(self.pts[t+1][1] + os_y)),
                                                        (0, 255, 0), 3)
                                        for t in range(13, 16):
                                            if t+1 < len(self.pts):
                                                cv2.line(white,
                                                        (int(self.pts[t][0] + os_x), int(self.pts[t][1] + os_y)),
                                                        (int(self.pts[t+1][0] + os_x), int(self.pts[t+1][1] + os_y)),
                                                        (0, 255, 0), 3)
                                        for t in range(17, 20):
                                            if t+1 < len(self.pts):
                                                cv2.line(white,
                                                        (int(self.pts[t][0] + os_x), int(self.pts[t][1] + os_y)),
                                                        (int(self.pts[t+1][0] + os_x), int(self.pts[t+1][1] + os_y)),
                                                        (0, 255, 0), 3)
                                        
                                        # Palm connections on white
                                        if len(self.pts) > 17:
                                            cv2.line(white,
                                                    (int(self.pts[5][0] + os_x), int(self.pts[5][1] + os_y)),
                                                    (int(self.pts[9][0] + os_x), int(self.pts[9][1] + os_y)),
                                                    (0, 255, 0), 3)
                                            cv2.line(white,
                                                    (int(self.pts[9][0] + os_x), int(self.pts[9][1] + os_y)),
                                                    (int(self.pts[13][0] + os_x), int(self.pts[13][1] + os_y)),
                                                    (0, 255, 0), 3)
                                            cv2.line(white,
                                                    (int(self.pts[13][0] + os_x), int(self.pts[13][1] + os_y)),
                                                    (int(self.pts[17][0] + os_x), int(self.pts[17][1] + os_y)),
                                                    (0, 255, 0), 3)
                                            cv2.line(white,
                                                    (int(self.pts[0][0] + os_x), int(self.pts[0][1] + os_y)),
                                                    (int(self.pts[5][0] + os_x), int(self.pts[5][1] + os_y)),
                                                    (0, 255, 0), 3)
                                            cv2.line(white,
                                                    (int(self.pts[0][0] + os_x), int(self.pts[0][1] + os_y)),
                                                    (int(self.pts[17][0] + os_x), int(self.pts[17][1] + os_y)),
                                                    (0, 255, 0), 3)
                                        
                                        # Draw landmarks on white
                                        for i in range(len(self.pts)):
                                            cv2.circle(white,
                                                      (int(self.pts[i][0] + os_x), int(self.pts[i][1] + os_y)),
                                                      4, (0, 0, 255), -1)
                                        
                                        # Run prediction on the white background image
                                        self.predict(white)
                                        
                                        # Draw predicted character on both frames
                                        cv2.putText(cv2image, f"Pred: {self.current_symbol}", 
                                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   1, (0, 255, 0), 2, cv2.LINE_AA)
                                        cv2.putText(cv2image_with_landmarks, f"Pred: {self.current_symbol}", 
                                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Convert normal frame for Tkinter display
                rgb_image_normal = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(rgb_image_normal)
                imgtk_normal = ImageTk.PhotoImage(image=self.current_image)
                self.panel_normal.imgtk = imgtk_normal
                self.panel_normal.config(image=imgtk_normal)
                
                # Convert landmark frame for Tkinter display
                rgb_image_landmark = cv2.cvtColor(cv2image_with_landmarks, cv2.COLOR_BGR2RGB)
                self.current_image_with_landmarks = Image.fromarray(rgb_image_landmark)
                imgtk_landmark = ImageTk.PhotoImage(image=self.current_image_with_landmarks)
                self.panel_landmark.imgtk = imgtk_landmark
                self.panel_landmark.config(image=imgtk_landmark)
                
                # Update character and sentence displays
                self.panel3.config(text=self.current_symbol, font=("Courier", 30))
                self.panel5.config(text=self.str, font=("Courier", 20), wraplength=250)
                
        except Exception as e:
            print(f"Error in video_loop: {traceback.format_exc()}")
        finally:
            self.root.after(10, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def speak_fun(self):
        if self.str.strip():
            self.speak_engine.say(self.str)
            self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.current_symbol = " "

    def predict(self, test_image):
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][
                1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][
                0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < \
                    self.pts[20][0]:
                ch1 = 3

        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > \
                    self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][
                0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6

        # condition for [yj][x]
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]
        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][
                1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][
                1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                 self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] >
                    self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                 self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] <
                    self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                 self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
                    (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                     self.pts[16][1] and
                     self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                 self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][
                0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] >
                    self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]
        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][
                1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][
                0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][
                0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][
                1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][
                0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][
                1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][
                0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][
                1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][
                1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][
                1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = " "

        if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[4][0] < self.pts[5][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] >
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = "next"

        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and (
                    self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] <
                    self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (
                    self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] <
                    self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'

        if ch1 == "next" and self.prev_char != "next":
            if self.ten_prev_char[(self.count - 2) % 10] != "next":
                if self.ten_prev_char[(self.count - 2) % 10] == "Backspace":
                    self.str = self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count - 2) % 10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]

        if ch1 == "  " and self.prev_char != "  ":
            self.str = self.str + "  "

        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count % 10] = ch1

    def destructor(self):
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()