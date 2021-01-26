import cv2
import numpy as np
import time



def main():
    
    circle_list = np.empty((0,3), dtype = np.uint16)
    distance_threshold = 40

    capture = cv2.VideoCapture(0)

    


    if(capture.isOpened() == False):
        return -1


    while(True):
        ret,frame = capture.read()
        image = frame

        windowsize = (800,600)
        frame = cv2.resize(frame, windowsize)

        
        mask = np.zeros(windowsize, dtype=np.uint8)

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(img, 3)
        img = cv2.Laplacian(img, cv2.CV_8U, 3)

        cimg  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=20, maxRadius=30)

        
        

        #circles:新規(3次元list)
        #circle_list:もともと

        if (circles is not None):
            circles = np.array(circles)

            circles = circles.reshape([circles.shape[1], circles.shape[2]])

            circles = np.uint16(np.around(circles))

            

            for h in circle_list:
                for l in circles:
                    
                    a = np.array([h[0], h[1]])
                    b = np.array([l[0], l[1]])

                    distance = np.linalg.norm(a - b)

                    if(distance <= distance_threshold):
                        circles = np.delete(circles, np.where(circles == l)[0], axis=0)
            
            circle_list = np.append(circle_list, circles, axis=0)
                        




            for i in circles:#i[0]:円中心x座標,i[1]:円中心y座標,i[2]:半径

                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

                ROI = cv2.bitwise_and(image, mask)

                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                x,y,w,h = cv2.boundingRect(mask)

                result = ROI[y:y+h, x:x+w]

                result[mask == 0] = (255, 255, 255)



        cv2.imshow('circles', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""

img = cv2.imread("hanko3.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""