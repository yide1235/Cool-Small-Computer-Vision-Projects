import bpy
import mathutils
from mathutils import Vector
import bmesh
from bpy import context
from random import *
import math




#so in my code its something like this    C------D
#                                          -      -
#                                           -      -
#                                            B------A


# i used sum of two ruled face then minus one normal face,
# but i assume AB is parallel to x and BC is parallel to y
# so in this case the caculation will be easier instead of calculating the 3d ecliud length







#a function for reading the file
def make_Verts(file_path):
    vertices=[]
    
    for line in open(file_path, "r"):
        #go over each line
        if line.startswith('#'):continue
        values=line.split()
        
        if not values:continue
        if values[0]=='v':
            vertex=[]
            vertex.append(10*float(values[1]))
            vertex.append(10*float(values[2]))
            vertex.append(10*float(values[3]))
            vertices.append(vertex)
        
    

    return vertices


def double_verts(vertices):

    verticesReturn=[]
    vertices2=[]
   

    for vertex in vertices:
        vertex2=[]
        vertex2.append(vertex[0])
        vertex2.append(vertex[1])
        vertex2.append(vertex[2]+1.5)
        vertices2.append(vertex2)
    verticesReturn.extend(vertices2)
    return verticesReturn



def fill_with_bezier(vertices,rate):
    points=vertices
    if len(points) == 1:
        return points

    left = points[0]
    ans = []
    for i in range(1, len(points)): 
        right = points[i]
        disX = right[0] - left[0]
        disY = right[1] - left[1]

        nowX = left[0] + disX * rate
        nowY = left[1] + disY * rate
        ans.append([nowX, nowY])

       
        left = right

    return fill_with_bezier(ans, rate)



def fill_with_bezier1(vertices,rate):
    
    X=[]
    Y=[]
    Z=[]
    for r in range(1,100):
        r=r/100
        a=fill_with_bezier(vertices,rate=r)
        x=a[0][0]
        y=a[0][1]
        Z.append([x,y])
 
    
    return Z 



#for parallel to x
def keypoint_insert1(vertices1,vertices2):
    #input:[a,b,c],[a2,b2,c2]
    
    
    
    #we will just input two points and 
    
    #include the end points
    return_verts=[]
    seg=4#keypoints)num are actually keypoints_num-1, this is the sgementation
    distance=10#change of z
    delta=(vertices1[0]-vertices2[0])/seg
    return_verts.append(vertices1)
   
    
    return_verts.append([vertices1[0]-delta, vertices1[1],vertices1[2]+distance*random()])
    return_verts.append([vertices1[0]-delta*2, vertices1[1],vertices1[2]])
    return_verts.append([vertices1[0]-delta*3, vertices1[1],vertices1[2]-distance*random()])
    
    return_verts.append(vertices2)
    
    
    #keypoints_num is how many keypoints between the end points is key_points_num-1
    #it was divided into 2 segments.think something like the sin function between 0-2pi
    #this is the helper method for fill_with_bezier
    #assume the four points from input are four points in the same plane 
    
    

    #keypoints should be between pointsA and pointsB

    
    
    
    
    #returning points
    return return_verts
    

#for parallel to x
def keypoint_insert2(vertices1,vertices2):
    #input:[a,b,c],[a2,b2,c2]
    
    
    
    #we will just input two points and 
    
    #include the end points
    return_verts=[]
    seg=4#keypoints)num are actually keypoints_num-1, this is the sgementation
    distance=10#change of z
    delta=(vertices1[0]-vertices2[0])/seg
    return_verts.append(vertices1)
   
    #this is why we should have 2 keypoints_insert
    return_verts.append([vertices1[0], vertices1[1]-delta,vertices1[2]-distance*random()])
    return_verts.append([vertices1[0], vertices1[1]-delta*2,vertices1[2]])
    return_verts.append([vertices1[0], vertices1[1]-delta*3,vertices1[2]+distance*random()])    
    return_verts.append(vertices2)
    
    
    #keypoints_num is how many keypoints between the end points is key_points_num-1
    #it was divided into 2 segments.think something like the sin function between 0-2pi
    #this is the helper method for fill_with_bezier
    #assume the four points from input are four points in the same plane 
    
    

    #keypoints should be between pointsA and pointsB

    
    
    
    
    #returning points
    return return_verts
    


# we need this two because bezier is for 2 dimension
def point_convert(verts,a,b):
    #3d to 2d
    #a,b are which axis we want
    
    result=[]
    for i in range(len(verts)):
        result.append([verts[i][a],verts[i][b]])
    
    
    return result


def point_convert_back(verts,a,value):
    #2d to 3d
    #a is the axis should be added
    #value is the value for the axis
    result=[]
    
    if a==0:
        
        for i in range(len(verts)):
            result.append([value, verts[i][0],verts[i][1]])
    
    
    if a==1:
        
        for i in range(len(verts)):
            result.append([ verts[i][0],value,verts[i][1]])
    
    
    if a==2:
        
        for i in range(len(verts)):
            result.append([verts[i][0],verts[i][1],value])
    
    return result




def generates(xy,xy1,length):
    
    result=[]
    #since for xy,xy1
    
    #we could discuss in different situation
    
    #only change of y
    
    if xy[0][0]==xy1[0][0]:
        deltax=(xy1[0][0]-xy1[length-1][0])/(length+1) #use for i
        deltay=(xy[0][1]-xy1[0][1])/(length+1)
        #they are parallel to x-axis
        #AB is xy, DC is xy1
        print('first')
        for i in range(length):
            for j in range(length):
               result.append([xy1[0][1]+deltay*j,xy1[0][0]-deltax*i,xy1[i][2]])
        print('\n')
    
    if xy[0][1]==xy1[0][1]:
        deltax=(xy[length-1][0]-xy1[length-1][0])/(length+1)
        deltay=(xy[0][1]-xy[length-1][1])/(length+1)
        print('second')
        #parallel to y-asix
        #AD is xy, BC is xy1
        for i in range(length):
            for j in range(length):
                result.append([xy1[length-1][0]-deltax*i,xy1[length-1][1]+deltay*j,xy[length-1-j][2]])
    
    return result
#result is all the points starts at D, then go to A for each iteration






def whole(pointsA,pointsB,pointsC,pointsD):


    #now top_verts are four points with height

    #i will use the top_verts because it has height 1.5,
    #so i could implement bezier on z direction.


    #because the input is 2d bezier curve so we need to choose where should be input





    AB=keypoint_insert1(pointsA,pointsB)
    #here AB is 5 points
    DC=keypoint_insert1(pointsD,pointsC)
    BC=keypoint_insert2(pointsB,pointsC)
    AD=keypoint_insert2(pointsA,pointsD)

    AB2=point_convert(AB,0,2)
    DC2=point_convert(DC,0,2)
    BC2=point_convert(BC,1,2)
    AD2=point_convert(AD,1,2)

    #before inputing the bezier points we need to finish the keypoints

    #because in my assume line AB is parallel to y-axis, so x we dont consider

    ABbezier=fill_with_bezier1(AB2,0.3)


    ABback=point_convert_back(ABbezier,1,pointsA[1])

    #add the two end point back
    ABtotal=[]
    ABtotal.append(pointsA)
    for i in range(len(ABback)):
        ABtotal.append(ABback[i])
    
    ABtotal.append(pointsB)
    #now ABtotal is 11 points , lets do the rest








    #for line between D and A
    #because for DC, we dont consider x,
    DCbezier=fill_with_bezier1(DC2,0.3)


    DCback=point_convert_back(DCbezier,1,pointsD[1])

    #add the two end point back
    DCtotal=[]
    DCtotal.append(pointsD)
    for i in range(len(DCback)):
        DCtotal.append(DCback[i])
    
    DCtotal.append(pointsC)



    #for line between B and C
    #for BC, we dont consider y
    BCbezier=fill_with_bezier1(BC2,0.8)


    BCback=point_convert_back(BCbezier,0,pointsB[0])

    #add the two end point back
    BCtotal=[]
    BCtotal.append(pointsB)
    for i in range(len(BCback)):
        BCtotal.append(BCback[i])
    
    BCtotal.append(pointsC)





    #for line between A and D
    #for AD, we dont consider y
    ADbezier=fill_with_bezier1(AD2,0.8)


    ADback=point_convert_back(ADbezier,0,pointsA[0])

    #add the two end point back
    ADtotal=[]
    ADtotal.append(pointsA)
    for i in range(len(ADback)):
        ADtotal.append(ADback[i])
    
    ADtotal.append(pointsD)




    #right now we have 4 lines
    #two pairs of line to calculate the ruled face
    #AB,DC formed ruled face
    #AD,BC formed ruled face

    #now lets generates points by order AD to BC, then we add the two ruled face
    #minus the original face, then we have the points for the mesh









    #lets start with AB,DC
    #collecting points in order AD to BC
    #now we know ABtotal is all the points with pointsA and pointsB, so
    length=len(ABtotal)

    #use length to generate points in the middle
    ABDC=generates(ABtotal,DCtotal,length)


   
    ABDC2=generates(ADtotal,BCtotal,length)
  


    #now we have two ruled faces, we need the plan face,
    #that is similar to project from ABDC to plannar z, so change all ABDC z-value to 0

    planxy=[]
    for i in range(length):
        planxy.append([ADtotal[i][0],ADtotal[i][1],1.5])
    
    planxy1=[]
    for i in range(length):
        planxy1.append([BCtotal[i][0],BCtotal[i][1],1.5])

    planner=[]
    planner=generates(planxy,planxy1,length)

  


    #now lets add two ruled faces and minus the planner face

    #now we have ABDC and ABDC2, add those two then minus planner this is for produce some randomness


    #final = ABDC+ABDC2-planner
    final=[]
    for i in range(length*length):
    
        final.append([ABDC[i][0]+ABDC2[i][0]-planner[i][0],
                        ABDC[i][1]+ABDC2[i][1]-planner[i][1],
                        ABDC[i][2]+ABDC2[i][2]-planner[i][2] ])
                   


    #now final1 is just a ruled faces, we need another, 

    return final




















file_path="C:\\Users\\myd97\\Desktop\\myd.txt"
#get the bottom four points
verts=make_Verts(file_path)
#get the top points
top_verts=double_verts(verts)
#now we have 8 points for only z difference with 1.5
#verts and top_verts
#print(verts)
#print('------')
#in my case the vertices with height are only those not with the inputs ones
#print(top_verts)
print('-------------------------')

pointsA=top_verts[0]
pointsB=top_verts[1]
pointsC=top_verts[2]
pointsD=top_verts[3]

final=[]
final=whole(pointsA,pointsB,pointsC,pointsD)













edges=[]

faces=[]

new_mesh=bpy.data.meshes.new('new_mesh')

new_mesh.from_pydata(final,edges,faces)
new_mesh.update()

new_object=bpy.data.objects.new('new_object',new_mesh)

new_collection = bpy.data.collections.new('new_collection')
bpy.context.scene.collection.children.link(new_collection)

new_collection.objects.link(new_object)






