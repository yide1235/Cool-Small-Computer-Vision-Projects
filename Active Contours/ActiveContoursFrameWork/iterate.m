function [newxt, newyt] = iterate(Ainv, xt, yt, Eext, gamma,kappa)

% Get fx and fy
[fx,fy]=gradient(Eext);

% Iterate


    
    newxt = Ainv*(gamma*xt- kappa*interp2(fx,xt,yt));
    newyt = Ainv*(gamma*yt- kappa*interp2(fy,xt,yt));
    



end

