#%%
import cv2
import imutils
import numpy as np


# %%

#url = 'rtsp://filipe:10203040@192.168.0.113:554'
url = 'http://192.168.0.25/media/?action=snapshot&user=admin&pwd='
#url = 0
cap = cv2.VideoCapture(url)
#back = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, img = cap.read()
    if type(img) == type(None):
        print('break image none')
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #fg = back.apply(img)
    #cv2.imshow('backsub',fg)
    cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('break key pressed')
        break
cap.release()
cv2.destroyAllWindows()
# %%
im = cv2.imread('http://192.168.0.25/media/?action=snapshot&user=admin&pwd=')
# %%
cv2.imshow(im)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
import requests
# %%
url = 'http://192.168.0.25/media/?action=snapshot&user=admin&pwd='
d = requests.get(url)

with open('oi.png', 'wb') as f:
    f.write(d.content)
# %%
from PIL import Image
from io import StringIO, BytesIO

r = requests.get(url)
for i in range(10):
    with BytesIO(r.content) as f:  
        i = Image.open(f)
        i.show()
# %%

# %%
