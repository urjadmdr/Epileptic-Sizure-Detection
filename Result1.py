from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import tkinter as tk
from tkinter import filedialog

import mysql.connector
from mysql.connector import Error
import time

from svm import svm_model

project_name = "Evaluation of EEG signal for detecting epilepsy (SEIZURE)"

from tkinter import Tk, Label, Entry, Toplevel, Canvas

from PIL import Image, ImageDraw, ImageTk, ImageFont

image = Image.open('SC.jpg')

import cv2
import numpy as np
import time
from keras.models import load_model

model = load_model('SEIZURE.h5')

import csv
import math
import statistics
from scipy.stats import *

import matplotlib.pyplot as plt
import pandas as pd

classes = ['SEIZURE Detected', 'Healthy Person']


########################################################################################
def get():
    login_page = Tk()
    login_page.geometry("1300x600+30+30")
    login_page.configure(background="#ffff8f")

    def LOGIN():

        def check_login():
            index = row_no.get()  # get entry
            print(index)
            index = int(index)

            ind = 0
            EEG = pd.read_csv('EEG.csv')
            with open('EEG.csv', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    ind = ind + 1
                    eeg = []
                    for i in range(0, 25):
                        eeg.append(float(row[i]))

                    if (ind == index):
                        break
            plt.plot(eeg)
            plt.ylabel('some numbers')
            plt.savefig('plot.png', dpi=50)

            img = cv2.imread('plot.png')
            img = cv2.resize(img, (488, 200))
            cv2.imwrite('plot.png', img)
            plt.clf()

            a = float(statistics.mean(eeg))
            b = float(statistics.pstdev(eeg))
            c = float(statistics.pvariance(eeg))
            d = float(statistics.stdev(eeg))
            e = float(statistics.variance(eeg))
            f = float(kurtosis(eeg))
            g = float(skew(eeg))
            h = min(eeg)
            i = max(eeg)

            minc = Image.open('plot.png')
            mincol = ImageTk.PhotoImage(minc)
            label4 = Label(login_page, image=mincol)
            label4.image = mincol
            label4.place(x=730, y=320, height=200, width=488)

            H2 = np.array([[a, b, c, d, e, f, g, h, i]], dtype=float)
            print(H2)

            ####
            predictions = svm_model.predict(H2)  #
            print(predictions)
            result_index = np.argmax(predictions[0])  #
            print(result_index)
            result = classes[result_index]  #
            print('Result : ', result)
            print(' ')
            ######

            label3 = Label(login_page, text=result)
            label3.configure(background="#ffffff")
            label3.config(font=("Times new roman", 15))
            label3.place(x=730, y=550, height=20, width=488)

        photoimage = ImageTk.PhotoImage(image)
        Label(login_page, image=photoimage).place(x=0, y=0)

        label2 = Label(login_page, text=project_name)
        label2.configure(background="#ffffFf")
        label2.config(font=("Courier", 20))
        label2.place(x=150, y=20, height=40, width=1000)

        label3 = Label(login_page, text="Enter Row number from dataset")
        label3.configure(background="#ffffff")
        label3.config(font=("Times new roman", 15))
        label3.place(x=700, y=250, height=25, width=300)

        row_no = StringVar()
        bank1Entry = Entry(login_page, textvariable=row_no)
        bank1Entry.configure(background="#ffffe0")
        bank1Entry.place(x=1030, y=250, height=25, width=70)

        B1 = Button(login_page, text="ENTER", command=check_login)
        B1.place(x=1150, y=250, height=25, width=75)
        B1.config(font=("T imes new roman", 15))
        B1.configure(background="#fffff0")

        login_page.mainloop()

    LOGIN()

