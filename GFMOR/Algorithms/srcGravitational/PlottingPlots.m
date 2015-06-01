%X1C1 = normrnd(1,0.5,100,1);
%X2C1 = normrnd(1,0.5,100,1);
%X1C2 = normrnd(7,0.5,100,1);
%X2C2 = normrnd(3,0.5,100,1);
%X2C3 = normrnd(5,0.5,100,1);
%X1C3 = normrnd(3,0.5,100,1);
%Y = [ones(100,1); ones(100,1)*2; ones(100,1)*3];
%X = [X1C1 X2C2; X1C2 X2C2; X1C3 X2C3];


figure(1)

distance = [0:0.001:1];
Force = 1./((distance+0.1).^(2));

d2 = (1-0.1*sqrt(1.734))/sqrt(1.734);
d1 = (1-0.1*sqrt(20.132))/sqrt(20.132);
d3 = (1-0.1*sqrt(23.006))/sqrt(23.006);


plot(distance,Force)
hold on
plot(d1,20.132,'*')
plot(d2,1.734,'*')
plot(d3,23.006,'*')

ylabel('Force')
xlabel('distance')
title('Euclidean Distance and Gravitational Force')
grid

figure(2)


WEuclidean = [0.5 0.99;0.0 0.9;0.3 0.9];

Weuclideanvector = reshape( WEuclidean.' ,1,numel(WEuclidean));
%c1Force = gravitation([1 5],1,X,Y,Weuclideanvector')
%c2Force = gravitation([1 5],2,X,Y,Weuclideanvector')
%c3Force = gravitation([1 5],3,X,Y,Weuclideanvector')
c1Force = 19.2547;
c2Force = 22.0308;
c3Force = 59.3527;

d2 = (1-0.1*sqrt(c2Force))/sqrt(c2Force);
d1 = (1-0.1*sqrt(c1Force))/sqrt(c1Force);
d3 = (1-0.1*sqrt(c3Force))/sqrt(c3Force);




plot(distance,Force);
hold on
plot(d1,c1Force,'*');
plot(d2,c2Force,'*');
plot(d3,c3Force,'*');

ylabel('Force');
xlabel('distance');
title('Weighted Euclidean Distance and Gravitational Force');
grid


figure(3)

WEuclidean = [1 1;1 1;1 1];
Weuclideanvector = reshape( WEuclidean.' ,1,numel(WEuclidean));
v = [3 1 1.75];

c1Force = gravitationRiccardi([1 5],1,X,Y,Weuclideanvector',v,100)
c2Force = gravitationRiccardi([1 5],2,X,Y,Weuclideanvector',v,100)
c3Force = gravitationRiccardi([1 5],3,X,Y,Weuclideanvector',v,100)

a1 = (1.0/100.0)^(1/v(1));
a2 = (1.0/100.0)^(1/v(2));
a3 = (1.0/100.0)^(1/v(3));
d1 = (1-a1*(c1Force.^(1.0/v(1))))/(c1Force.^(1.0/v(1)));
d2 = (1-a2*(c2Force.^(1.0/v(2))))/(c2Force.^(1.0/v(2)));
d3 = (1-a3*(c3Force.^(1.0/v(3))))/(c3Force.^(1.0/v(3)));

Force1 = 1./((distance+a1).^(v(1)));
Force2 = 1./((distance+a2).^(v(2)));
Force3 = 1./((distance+a3).^(v(3)));

plot(distance,Force1);
hold on
plot(distance,Force2);
plot(distance,Force3);
plot(d1,c1Force,'*');
plot(d2,c2Force,'*');
plot(d3,c3Force,'*');

ylabel('Force');
xlabel('distance');
title('Euclidean Distance and Generalized Force');
grid