#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image
import imutils
#%%
filename = 'captura/video017.mp4'
#pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'
# %%

cap = cv2.VideoCapture(filename)

cont = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imwrite('fotos/frame'+str(cont)+'.jpg', frame)
    cont += 1

# %%
cap.release()
cv2.destroyAllWindows()
# %%
img = cv2.imread('fotos/frame.jpg', cv2.IMREAD_GRAYSCALE )
newImg = cv2.blur(img,(5,5))
img    = newImg

laplacian   = cv2.Laplacian(img,cv2.CV_64F)
sobelx      = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
sobely      = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)

#aplicado o threshold sobre o Sobel de X
tmp, imgThs = cv2.threshold(sobelx,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)

#pequena chacoalhada nos pixels pra ver o que cai (isso limpa a img mas
#distancia as regioes, experimente)
#krl      = np.ones((6,6),np.uint8)
#erosion  = cv2.erode(imgThs,krl,iterations = 1)
#krl      = np.ones((19,19),np.uint8)
#dilation = cv2.dilate(erosion,krl,iterations = 1) 
#imgThs   = dilation

#estrutura proporcional aa placa
morph       = cv2.getStructuringElement(cv2.MORPH_RECT,(40,13))

#captura das regioes que possam conter a placa
plateDetect = cv2.morphologyEx(imgThs,cv2.MORPH_CLOSE,morph)
regionPlate = plateDetect.copy()

contours, hierarchy = cv2.findContours(regionPlate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cv2.drawContours(regionPlate,contours,-1,(255,255,255),18)
  
plt.subplot(3,3,1)
plt.imshow(img,cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,2)
plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian')
plt.xticks([]), plt.yticks([])

plt.subplot(3,3,3)
plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X')
plt.xticks([]), plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

plt.subplot(3,3,5)
plt.imshow(imgThs,cmap = 'gray')
plt.title('Threshold')
plt.xticks([]), plt.yticks([])

plt.subplot(3,3,6)
plt.imshow(plateDetect,cmap = 'gray')
plt.title('Morphology')
plt.xticks([]), plt.yticks([])

plt.subplot(3,3,7)
plt.imshow(regionPlate,cmap = 'gray')
plt.title('Draw Contours')
plt.xticks([]), plt.yticks([])

plt.show()
# %%
img.shape
# %%
img = cv2.imread('fotos/frame.jpg', cv2.IMREAD_GRAYSCALE )
crop_img = img[320:380, 500:590]
plt.imshow(crop_img, cmap='gray')
cv2.waitKey(0)
# %%

cv2.imshow('Image', crop_img)
cv2.waitKey(0) 
# %%
cap.release()
cv2.destroyAllWindows()
# %%
img = cv2.imread('fotos/frame.jpg', cv2.IMREAD_GRAYSCALE )
img = img[320:380, 500:590]
scale_percent = 200 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#img = cv2.medianBlur(img, 3)
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
light = cv2.morphologyEx(img, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Light Regions", light)
cv2.waitKey(0)  
cv2.destroyAllWindows()
# %%
cv2.imwrite('placa.png',img)
# %%
img = cv2.imread('carro.jpg', cv2.IMREAD_UNCHANGED )
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

width = int(img.shape[1])
height = int(img.shape[0])
dim = (int(width/10), int(height/10))
print(dim)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("img", img)
cv2.waitKey(0)  
cv2.destroyAllWindows()

img = cv2.medianBlur(img, 3)
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
light = cv2.morphologyEx(img, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
print('placa',pytesseract.image_to_string(light, config='--psm 13'))
cv2.imshow("Light Regions", light)
cv2.waitKey(0)  
cv2.destroyAllWindows()

# %%

img = cv2.imread('carro.jpg', cv2.IMREAD_UNCHANGED)

width = int(img.shape[1])
height = int(img.shape[0])
dim = (int(width/10), int(height/10))
print(dim)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray = cv2.bilateralFilter(gray, 13, 15, 15) 
gray = cv2.medianBlur(gray, 3)
light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
edged = cv2.Canny(light, 30, 200) 


contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours:
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

text = pytesseract.image_to_string(Cropped,  config='--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') #11
print("programming_fever's License Plate Recognition\n")
print("Detected license plate Number is:",text)
img = cv2.resize(img,(500,300))
Cropped = cv2.resize(Cropped,(400,200))
cv2.imshow('car',img)
cv2.imshow('Cropped',Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
