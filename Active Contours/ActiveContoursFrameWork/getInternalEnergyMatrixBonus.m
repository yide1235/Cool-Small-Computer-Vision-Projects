function [Ainv] = getInternalEnergyMatrixBonus( Eext, xt, alpha, beta, gamma,kappa)
%this function return Ainv

[m,arb] = size(xt);

a(1)=beta;
a(2)=-(alpha + 4*beta);
a(3)=(2*alpha + 6*beta);

%use cricshift
A = a(1)*circshift(eye(m),2);
A = A + a(2)*circshift(eye(m),1);
A = A + a(3)*circshift(eye(m),0);
A = A + a(2)*circshift(eye(m),-1);
A = A + a(1)*circshift(eye(m),-2);

%use [l u]
[l u] = lu(A+gamma.* eye(m));
Ainv = inv(u)* inv(l);
%return Ainverse

end

