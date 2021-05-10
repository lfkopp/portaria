#%%
import cv2
# %%
url = 'rtsp://filipe:10203040@192.168.0.113:554'
#url = 'rtsp://filipe:10203040@179.218.229.168:5005'
camera = cv2.VideoCapture(url)
width= int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,height)
writer= cv2.VideoWriter('captura/video020.mp4', cv2.VideoWriter_fourcc(*'X264'), 20, (width,height))

while(True):
    
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    writer.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
writer.release()
cv2.destroyAllWindows()
# %%

camera.release()
writer.release()
cv2.destroyAllWindows()

# %%

# %%
