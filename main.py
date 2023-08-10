'''
import cv2
import numpy as np
img =cv2.imread('images/IMG_0249.png')
new_img=cv2.resize(img, (300, 500)) # новая картинка
#cv2.imshow('Result', new_img)
#cv2.imshow('Result', img[0:100, 0:150]) # обрезание картинки
#img=cv2.GaussianBlur(img, (9, 9), 0) размытие

img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # серый слой
img=cv2.Canny(img, 90, 90)# бинарная картинка  в черном и белом
kernel = np.ones((5, 5), np.uint8)
img=cv2.dilate(img, kernel, iterations=1 ) # жирность обводки
img=cv2.erode(img, kernel, iterations=1) #оптимизаия работы с картинкой
cv2.imshow('Result', img)
print(img.shape) # ирина высота количество слоев
cv2.waitKey(0) # картинка показывается бесконечно
'''



''' Работа  с  видео
cap = cv2.VideoCapture('video/IMG_9941.MP4')
cap.set(3, 500)
cap.set(4, 300)
while True:
    success, img=cap.read()
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
'''
# введение





'''
import cv2
import numpy as np
#img =cv2.imread('images/IMG_0249.png')
cap = cv2.VideoCapture("video/IMG_9941.MP4")
while True:
    success, img=cap.read()

   # new_img = cv2.resize(img, (300, 500))  # новая картинка
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # серый слой
    img = cv2.Canny(img, 30, 30)  # бинарная картинка  в черном и белом
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)  # жирность обводки
    img = cv2.erode(img, kernel, iterations=1)  # оптимизаия работы с картинкой
    cv2.imshow('Result', img)


    #cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imshow('Result', img)
print(img.shape) # ирина высота количество слоев
cv2.waitKey(0) # картинка показывается бесконечно
'''




'''
import cv2
import numpy as np
img=cv2.imread("images/IMG_0249.png")
#img=cv2.flip(img, 1)
def rotate(img_param, angle): # поворот картинки
    height, width =img_param.shape[:2]
    point=(width // 2, height // 2)
    mat = cv2.getRotationMatrix2D(point, angle, 1)
    return cv2.warpAffine(img_param, mat, (width, height))
#img=rotate(img, 90)
def transform(img_param, x, y ):
    mat=np.float32([[1, 0, x], [0,1, y]])
    return cv2.warpAffine(img_param, mat, (img_param.shape[1], img_param.shape([0])))
#img= transform(img, 30, 200)
cv2.imshow('Result', img)
cv2.waitKey(0)
'''









'''
import cv2
import numpy as np
img=cv2.imread("images/IMG_0249.png")
new_img=np.zeros(img.shape, dtype='uint8' )

img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # сервый формат
img=cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.Canny(img, 100, 140)
con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(new_img, con, -1, (230, 111, 148), 1 ) # разрисовали пиксели
print(con) # получаем контуры
cv2.imshow("result", new_img)
cv2.waitKey(0)
'''


















'''
import cv2
import numpy as np
img=cv2.imread("images/IMG_0249.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img=cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
cv2.imshow("result", img)
cv2.waitKey(0)
'''







'''
import cv2
import numpy as np
img=np.zeros((350, 350), dtype='uint8')
#img=cv2.imread("images/IMG_0249.png")
square =cv2.rectangle(img.copy(), (25,25), (350, 350), 255, -1)
circle=cv2.circle(img.copy(), (0, 0), 80, 255, cv2.FILLED) # квадрат и круг
img = cv2.bitwise_and(circle, square) # находим общие черты и отображаем их
cv2.imshow("result", img)
cv2.waitKey(0)
'''






'''
import cv2
img = cv2.imread('images/2153351.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cv2.CascadeClassifier('faces.xml')
results = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=1)
for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
cv2.imshow("result", img)
cv2.waitKey(0)
'''













'''
import cv2

# Load the images
#img = cv2.imread('images/2153351.jpg')
img = cv2.imread('images/IMG_0253.jpg')
replace_img = cv2.imread('images/IMG_0255.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the pre-trained face detection classifier
faces = cv2.CascadeClassifier('faces.xml')

# Detect faces in the grayscale image
results = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Replace faces with the new image
for (x, y, w, h) in results:
    replace_face = cv2.resize(replace_img, (w, h))
    img[y:y+h, x:x+w] = replace_face

# Display the modified image
cv2.imshow("Result", img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


import cv2

# Load the images
img = cv2.imread('images/IMG_0253.jpg')
replace_img = cv2.imread('images/IMG_0255.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the pre-trained face detection classifier
faces = cv2.CascadeClassifier('faces.xml')

# Detect faces in the grayscale image
results = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Replace faces with the visible part of the new image
for (x, y, w, h) in results:
    replace_face = cv2.resize(replace_img, (w, h))
    img_height, img_width, _ = img.shape
    x1, x2 = max(0, x), min(img_width, x + w)
    y1, y2 = max(0, y), min(img_height, y + h)

    replace_face = replace_face[0:y2 - y1, 0:x2 - x1]  # Crop to visible part
    img[y1:y2, x1:x2] = replace_face

# Display the modified image
cv2.imshow("Result", img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml