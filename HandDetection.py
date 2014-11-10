import cv2
import numpy as np

#Open Camera object
cap = cv2.VideoCapture(0)

def nothing(x):
    pass

# Function to find angle between 
def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

def FindDistance(A,B): 
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2)) 
 

# Creating a window for later use
cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)

while(1):

    ret, frame = cap.read()
    blur = cv2.blur(frame,(3,3))
 	#converting to HSV
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
 	# get info from track bar and apply to result
    h = cv2.getTrackbarPos('h','HSV_TrackBar')
    s = cv2.getTrackbarPos('s','HSV_TrackBar')
    v = cv2.getTrackbarPos('v','HSV_TrackBar')

	# Normal masking algorithm
    lower = np.array([h,s,v])
    upper = np.array([180,255,255])

    mask = cv2.inRange(hsv,lower, upper)
    
    #hard coded Orange valuess
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    #Original HSV filtered image
    #result = cv2.bitwise_and(frame,frame,mask = mask)
    
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #Increase the orange area
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    #Decrease noise
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)
    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)

    dilation2 = dilation2 + filtered + erosion + dilation + dilation3
    
    median = cv2.medianBlur(dilation2,5)
    
    
    ret,thresh = cv2.threshold(median,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
	#Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0	
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i  
				  
    cnts = contours[ci]
    hull = cv2.convexHull(cnts)
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)
    FarDefect = []
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
        cv2.line(frame,start,end,[255,0,0],1)
        cv2.circle(frame,far,10,[255,255,255],3)
    
    #print '#####'
    #print FarDefect[0]
    #print '#####'
    #print len(FarDefect)

     
     
    
	#Draw center of the hand
    moments = cv2.moments(cnts)
    if moments['m00']!=0:
        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
    centerMass=(cx,cy)       
    cv2.circle(frame,centerMass,7,[100,0,255],2)     
    
    distanceFromDefects = []
    
    #for i in range(0,len(FarDefect)):
    #    x =  np.array(FarDefect[i])
    #    centerMass = np.array(centerMass)
   #     #print np.sqrt(np.power(x[0]-centerMass[0],2)-np.power(x[1]-centerMass[1],2)
    #distanceFromDefects.append(np.sqrt(np.power(x[0]-centerMass[0],2)-np.power(x[1]-centerMass[1],2))
    #print distanceFromDefects
    # centerMass = np.asarray(centerMass)

	#distanceFromDefects.append(FindDistance(centerMass,Defect))
    
    #print 'original',contours[0][1]
    #print contours[0][2][0][0]
    #print contours[0][2][0][1]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Center',centerMass,font,2,(255,255,255),2)
    
    finger = []
    for i in range(0,len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 30) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 30):
            finger.append(hull[i][0])
    
    #Sort fingers by hight  
    finger =  sorted(finger,key=lambda x: x[1])   
    
    
    #show height raised fingers
    #cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)

    #Print angles between fingers
    center = np.array(centerMass)
    finger[0] = np.array(finger[0])
    finger[1] = np.array(finger[1])
    v1 = finger[0] - center
    v2 = finger[1] - center
    angle = Angle(v1,v2)
    cv2.putText(frame,str(angle),(100,100),font,2,(255,255,255),2)
        
    ### Bounding rectangle ####
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.drawContours(frame,[img],0,(0,0,255),2)
    
	#cv2.drawContours(frame,contours,-1,(200,50,100),2)
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
	#cv2.circle(frame,[defects],5,[0,0,255],-1)

    cv2.imshow('Dilation',frame)

    

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
