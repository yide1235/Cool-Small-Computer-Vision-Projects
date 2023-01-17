'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from msilib.schema import IniFile
from _ssl import nid2obj

parser = ArgumentParser()
#0 is using 8-points
#1 is using ransac
parser.add_argument("--UseRANSAC", type=int, default=1)
parser.add_argument("--image1", type=str,  default='data/m1.jpg' )
parser.add_argument("--image2", type=str,  default='data/m2.jpg' )
args = parser.parse_args()

print(args)


def FM_by_normalized_8_point(pts1,  pts2):
    #F, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )
    # comment out the above line of code. 
    
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 
    
    add1 = np.ones(pts1.shape[0]).reshape(pts1.shape[0], 1)

    pts1 = np.hstack((pts1, add1))
    pts2 = np.hstack((pts2, add1))
    pts1 = pts1.T
    pts2 = pts2.T

    # normalize image coordinates
    mean1 = np.mean(pts1[:2], axis=1)
    N1 = np.array([[np.sqrt(2) / np.std(pts1[:2]), 0, -np.sqrt(2) / np.std(pts1[:2]) * mean1[0]],
               [0, np.sqrt(2) / np.std(pts1[:2]), -np.sqrt(2) / np.std(pts1[:2]) * mean1[1]], [0, 0, 1]])
    pts1 = np.dot(N1, pts1)


    mean2 = np.mean(pts2[:2], axis=1)
    N2 = np.array([[np.sqrt(2) / np.std(pts2[:2]), 0, -np.sqrt(2) / np.std(pts2[:2]) * mean2[0]],
               [0, np.sqrt(2) / np.std(pts2[:2]), -np.sqrt(2) / np.std(pts2[:2]) * mean2[1]], [0, 0, 1]])
    pts2 = np.dot(N2, pts2)

    # compute F with the normalized coordinates

    A = np.zeros((pts1.shape[1], 9))
    for i in range(pts1.shape[1]):
        A[i] = [pts1[0, i] * pts2[0, i], pts1[0, i] * pts2[1, i], pts1[0, i] * pts2[2, i],
            pts1[1, i] * pts2[0, i], pts1[1, i] * pts2[1, i], pts1[1, i] * pts2[2, i],
            pts1[2, i] * pts2[0, i], pts1[2, i] * pts2[1, i], pts1[2, i] * pts2[2, i]]


    E=np.dot(A.T , A)
    
    
    #i commented this code out because this code does not give the least error
    '''
    w,v= np.linalg.eig(E)

    
    leastEigValue=w.min()

    np.fill_diagonal(E,E.diagonal()-leastEigValue)

    #now we plug the smallest eignvalue to get the eigenvector
    b=np.array([1,1,1,1,1,1,1,1,1])
    b=b.T


    #x=np.linalg.solve(E,b)
    #x=np.linalg.inv(E).dot(b)
    #since we are using float to calculate so find if Ax-b is close to 0
    # print("1111111", E@x-b)
    
    
    #for this part, its works perfectly for the first part, but for ransac, it will produce error,
    #so i changed a way to find the eigen vector with the least eigenvalue.
    '''
    
    #use eigh because it gives eigenvector by order
    w1,v1=np.linalg.eigh(E)
    #print(w1,v1)
    
    #print(x)
    #print(v1[:,0])
    
    F = v1[:,0].reshape(3, 3).copy()
    

    F=F.T

    #for this part , i compare F with F i calculated and see the error
    #U, S, V=np.linalg.svd(F)

    #S[2]=0
    #S=np.diag(S)
    #F1=np.dot(U,np.dot(S,V))
    #should be using a single svd, but we are not allowed
    # so use eig to calculate U and V then calcualte Sig,
    #then use division
    w1,v1=np.linalg.eig(np.dot(F.T,F))
    w,u=np.linalg.eig(np.dot(F,F.T))
 
    
    #remeber u @ sig @ v= F
    #w should be equal to y
    #w=np.sort(w)
    #w=np.flip(w)
    

    w=np.abs(w)
    #s is just for calculating v
    s=np.sqrt(w)
   
    
    
    
    s=np.diag(s)
    
    
    
 
    #so far s is totally correct.
    #-----------------
    
    temp=np.dot(u,s)
    temp=np.linalg.pinv(temp)
  
  
    
    #s1=np.linalg.inv(s)
    #u1=u.T
    #actually is v.T
    v= temp.dot(F)
    
    s=np.diag(s)
    s=s.copy()
    s[np.argmin(s)]=0
    s=np.diag(s)
    
    
    #also i commented this part out because 
    '''
    #print(np.dot(s1,u1))
    s=np.diag(s)
    s2=np.copy(s)
    print(s2)
    print(np.argmin(s2))
    s2[0]=0
    
    s2=np.diag(s2)
    #print(w)
    print(u,U)
    '''


    #use eigen to recover elements for calculating svd
    #F= np.dot(u,np.dot(s,v))
    

    
    F=u @ s @ v

    # denormalization
    F = N2.T @ F @ N1


    
    # F:  fundmental matrix
    return  F/F[2,2]




def FM_by_RANSAC(pts1,  pts2):
    #F2, mask = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )	
    #initialize mask
    mask1=np.zeros((pts1.shape[0]))
    mask1=np.uint8(mask1)
    mask1=mask1.reshape(pts1.shape[0],1)
    # comment out the above line of code. 
 
    add1 = np.ones(pts1.shape[0]).reshape(pts1.shape[0], 1)

    pts11 = np.hstack((pts1, add1))
    pts22 = np.hstack((pts2, add1))


    
    #initializes
    n=0
    ni=0
    F=np.zeros([3,3])
    inlierPoints=[]
    iteration=1000
    confidence=0.06
    
    #mask1 is initialized mask
 
    #now x2 are corresponding coord for x1
    
    for i in range(iteration):        
        

        
        #initialize 8 pints first
        index_array=np.zeros((pts1.shape[0]))
    
        for i in range(pts1.shape[0]):
            index_array[i]=i
    
        index=np.zeros(8)
        index=np.random.choice(index_array,8)
            #now index full with 8 indexes
    
        x1=np.zeros((8,2))
        x2=np.zeros((8,2))
    
    
    
        for i in range(8):
            x1[i]=pts1[int(index[i])]
            x2[i]=pts2[int(index[i])]
        
        #now we have random 8 corresponding points   

        Fi=FM_by_normalized_8_point(x1,  x2)
        
        
      
        add = np.ones(x1.shape[0]).reshape(x1.shape[0], 1)
        x1=np.hstack((x1,add))
        x2=np.hstack((x2,add))
        
   
        result=np.zeros(len(pts11)).reshape(len(pts11),1)


        #test if xFx' is equal to 0
        for i in range(len(pts22)):
            
            temp=np.dot(Fi, pts11.T[:,i])
            
            result[i]=np.dot(pts22[i,:],temp.T)
            result[i]=np.abs(result[i])
            if abs(result[i])<confidence:
                
                mask1[i]=1
            else:
                mask1[i]=0
                
        
        ni=np.sum(mask1)
        
        if ni> n:
            n=ni
            F=Fi
            
             
        
            
                
       

    
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 
 
    # F:  fundmental matrix
    # mask:   whetheter the points are inliers
   
    #print(np.sum(np.abs(F2-F))) 
    return  F,mask1

	
img1 = cv2.imread(args.image1,0) 
img2 = cv2.imread(args.image2,0)  

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
		
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
if args.UseRANSAC:
    F,  mask = FM_by_RANSAC(pts1,  pts2)
    
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]	
else:
    F = FM_by_normalized_8_point(pts1,  pts2)
	

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
	
	
# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img6)
plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img4)
plt.subplot(122),plt.imshow(img3)
plt.show()
