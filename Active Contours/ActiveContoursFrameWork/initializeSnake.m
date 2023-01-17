function [xt,yt] = initializeSnake(I)

% Get initial points
x=[];y=[];xt=[];yt=[];
%they are both arrays
i=1;N=50;

while i<N

    [xi,yi,button] = ginput(1); 
    
    %increase the size
    x(i)=xi; 
    y(i)=yi;

    hold on;
    %for showing points in the pic
    plot(xi,yi,'ro');
   
    %for ginput, the right click is button==3 
    if  button == 3
        break; 
    end
    i = i+1;
end

store = [x;y];
length = i+1;%for connecting to last one


%connect last to first
store(:,length) = store(:,1);

% Interpolate
first = 1:length;
second = 1:0.04:length;
%spacing is 0.04

inter = spline(first,store,second);
%the first one
xt=inter(1,:); 
%the second one
yt=inter(2,:); 

% Clamp points to be inside of image

temp = plot(x(1),y(1),'ro',xt,yt,'b.');

xt=xt';%change to transpose
yt=yt';

%xt',yt' are for return

end

