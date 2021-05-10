#%%
import cv2
import imutils
import numpy as np
import pytesseract
import glob
from time import sleep

# %%

cont = 8410
url = 'rtsp://filipe:10203040@192.168.0.113:554'
cap = cv2.VideoCapture(url)
car_cascade = cv2.CascadeClassifier('cascades/cars.xml')
back = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
print('iniciando')
while True:
    ret, img = cap.read()
    if type(img) == type(None):
        #break
        print('sem imagem')
        sleep(100)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = gray.copy()
        gray = back.apply(img)
        #     _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        #      contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     for cnt in contours:
        # # Calculate area and remove small elements
        # area = cv2.contourArea(cnt)
        # if area > 100:
        #     cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
        #     gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        #results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.dilate(gray, kernel, iterations=10)
        gray = cv2.bitwise_and(gray2, gray)
        #gray = gray[:,:1000]
        cars = car_cascade.detectMultiScale(gray, 1.05,  minNeighbors=2, minSize=(120,120))
        cv2.imshow('video',gray)

        for (x, y, w, h) in cars:
            x2,y2 = gray.shape
            k=100
            x = max(0,x-k)
            #y = max(0,y-k)
            w = min(w+2*k,x2-x)
            h = min(h+k,y2-y)
            cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
            cv2.rectangle(img,(max(0,x-100),y),(x+w,min(y2,y+h+200)),(0,127,255),2)
            roi_gray = gray2[y:y+h, x:x+w]
            cv2.imwrite('carros/carro'+str(cont)+'.png',roi_gray)
            cont += 1

    except:
        print('.',end='')
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# %%
cap.release()
cv2.destroyAllWindows()
# %%
cars = glob.glob('carros/pos/*.png')
#cars= ['carros/carro4283.png']
for car_file in cars:
    img = cv2.imread(car_file, cv2.IMREAD_GRAYSCALE)
    
    #print(img.shape)
    img = cv2.resize(img,(1000,1000))
    img2 = img.copy()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(img, 13, 15, 15) 
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    krl      = np.ones((4,4),np.uint8)
    light  = cv2.erode(gray,krl,iterations = 1)
    dilation = cv2.dilate(light,krl,iterations = 1) 
    edged = cv2.Canny(light, 14, 30) 

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:20]
    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            k = 2
            approx[0][0][0] += k
            approx[0][0][1] -= k
            approx[1][0][0] -= k
            approx[1][0][1] -= k
            approx[2][0][0] -= k
            approx[2][0][1] += k
            approx[3][0][0] += k
            approx[3][0][1] += k

            screenCnt = approx
            break
    if screenCnt is None:
        detected = 0
        #print ("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
        #cv2.imshow('img',img)
        #cv2.imshow('img2',img2)


        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (max(0,np.min(x)), max(0,(np.min(y))))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = img2[topx:bottomx+1, topy:bottomy+1]
        Cropped = cv2.resize(Cropped,(0,0), fx=2,fy=2, interpolation = cv2.INTER_CUBIC)
        Cropped = cv2.GaussianBlur(Cropped,(3,3),cv2.BORDER_DEFAULT)
        Cropped = cv2.adaptiveThreshold(Cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        
        #cv2.imshow('Cropped',Cropped)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print('shape',Cropped.shape)
        hh,ww = Cropped.shape
        if (ww>=2*hh) and (ww<3.5*hh):
            for psm in range(3,14):
                text = pytesseract.image_to_string(Cropped,  config=f'--psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') #11
                text = str(text).strip()
                text = text.replace(' ','')
                text = text.replace('\n','')
                print(psm,"len:",len(text),"Placa:",text,'shape',ww,hh)
                if len(text) > 3:
                    placa_file = car_file.replace('carro','placa')
                    placa_file = placa_file.replace('.','-'+str(psm)+'-'+str(text)+'.')
                    #print('placa_file',placa_file)
                    cv2.imwrite(placa_file,img2[topx:bottomx+1, topy:bottomy+1])

# %%

cv2.destroyAllWindows()
# %%
