I1 = imread('sample.jpg');

ITERATIONS = 10;

I = im2double(I1);
t = cputime;

for j=1:ITERATIONS,
  pedro = features_pedro(I, 8);
end
time1 = cputime-t;


I = im2single(I1);
t = cputime;
for j=1:ITERATIONS,
  mmex = features_madmex(I, 8);
end
time2 = cputime-t;

disp(sprintf('Sequential solution %f', time1));
disp(sprintf('Madmex parallel solution %f with speedup %0.4fx', time2, time1/time2));

c=zeros(length(pedro),1);

for i=1:numel(pedro)
   if abs(pedro(i) - mmex(i)) < .1
     c(i)=0;
   else
     c(i)=1;
     %display(sprintf('us: %f ref: %f',mmex(i), pedro(i)));
   end
end

err = sum(c(:));

display(sprintf('off: %d percent deviation: %f',err, 100.0*err/(134*length(pedro))))


quit();
%disp(o);

