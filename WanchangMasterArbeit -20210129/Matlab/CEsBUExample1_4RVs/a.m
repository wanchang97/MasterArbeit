function a = a(x,t)
aPart1 = (1-x(4,:)./2).*exp(x(3,:)).*x(2,:).^x(4,:).*pi.^(x(4,:)./2).*1e5.*t;
aPart2 = x(1,:).^(1-x(4,:)./2);
aPart3 = (1-x(4,:)./2).^(-1);
%a = power(aPart1+aPart2,aPart3);

%a = abs(real(power(aPart1+aPart2,aPart3)));
a = real(power(aPart1+aPart2,aPart3));
end

% function a = a(x,t)
% global a0 DS
% aPart1 = (1-x(2,:)./2).*exp(x(1,:)).*DS.^x(2,:).*pi.^(x(2,:)./2).*1e5.*t;
% aPart2 = a0.^(1-x(2,:)./2);
% aPart3 = (1-x(2,:)./2).^(-1);
% %a = power(aPart1+aPart2,aPart3);
% 
% %a = abs(real(power(aPart1+aPart2,aPart3)));
% a = real(power(aPart1+aPart2,aPart3));
% end


