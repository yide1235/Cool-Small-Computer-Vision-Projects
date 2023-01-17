
function [Eext] = getExternalEnergy(I_smooth,Wline,Wedge,Wterm)

Eline = repmat(I_smooth,1);  

[gradx,grady]=gradient(I_smooth);  
Eedge = -sqrt((gradx.*gradx+grady.*grady));  



a1 = [-1 1];   
a2 = [-1;1];  
a3 = [1 -2 1];   
a4 = [1;-2;1];  
a5 = [1 -1;-1 1];  


cx = conv2(I_smooth,a1,'same');  
cy = conv2(I_smooth,a2,'same');  
cxx = conv2(I_smooth,a3,'same');  
cyy = conv2(I_smooth,a4,'same');  
cxy = conv2(I_smooth,a5,'same');  


[row,col]=size(I_smooth);
Eterm=[row,col];

for i = 1:row  
    for j= 1:col  
        a= (cyy(i,j)*cx(i,j)*cx(i,j) -2 *cxy(i,j)*cx(i,j)*cy(i,j) + cxx(i,j)*cy(i,j)*cy(i,j));
        b=((1+cx(i,j)*cx(i,j) + cy(i,j)*cy(i,j))^1.5);
        Eterm(i,j) = a/b;
    end  
end  
  

Eext = Wline*Eline + Wedge*Eedge + Wterm*Eterm;  
  
end

