import matplotlib.pyplot as plt
import Yolov5ForCSGO.aim_csgo.grabscreen as grabscreen
import time

time.sleep(5)
img = grabscreen(region=(0,0,1920,1080))

plt.figure()

plt.plot(img)

plt.show()