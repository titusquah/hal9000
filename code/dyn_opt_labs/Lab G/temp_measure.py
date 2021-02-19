# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:23:24 2019

@author: Titus
"""

from tclab import TCLab
from time import sleep
from pygame import mixer

a = TCLab()
while True:
  try:
    print(a.T1,a.T2)
    if a.T1<22.5:
      a.Q1(30)
    if a.T1<24 and a.T1>23:
      a.Q1(0)
      mixer.init()
      mixer.music.load(r"C:\Users\tq220\Music\Aimer\HDAfter Dark - Aimer - ポラリス Polaris中日字幕.mp3")
      mixer.music.play() 
    sleep(3)
  except KeyboardInterrupt:
    print("Exiting board")
    a.Q1(0)
    a.Q2(0)
    a.LED(0)
    a.close()
    
    mixer.music.stop()
    break