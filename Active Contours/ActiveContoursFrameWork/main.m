clear all;

% Parameters (play around with different images and different parameters)
N = 2000;
alpha = 0.2;
beta = 0.2;
gamma = 1;
kappa = 0.1;
Wline = 0.1;
Wedge = 0.4;
Wterm = 0.1;
sigma = 1.0;

% Load image
I = imread('images/brain.png');
%this if transfer colorful to grey
if (ndims(I) == 3)
    I = rgb2gray(I);
end


% Initialize the snake
I_smooth = double(imgaussfilt(I, sigma));
figure(2),imshow(I);
[xt,yt]=initializeSnake(I_smooth);



% Calculate external energy
Eext=getExternalEnergy(I_smooth,Wline,Wedge,Wterm);


% Calculate matrix A^-1 for the iteration
Ainv = getInternalEnergyMatrixBonus(Eext,xt, alpha, beta, gamma,kappa);
% Iterate and update positions
figure(3)
% Iterate and update positions
displaySteps = floor(N/10);
for i=1:N
% Iterate

[xt,yt] = iterate(Ainv, xt, yt, Eext, gamma,kappa);
    imshow(I);   
    hold on;  
    plot([xt; xt(1)], [yt; yt(1)], 'r');  
    

        % Display step
    if(mod(i,displaySteps)==0)
        fprintf('%d/%d iterations\n',i,N);
    end
    
    pause(0.001) 


end


if(displaySteps ~= N)
    fprintf('%d/%d iterations\n',N,N);
end



