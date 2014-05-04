I1 = imread('sample.jpg');

ITERATIONS = 10;

% STANDARD DOUBLE
I = im2double(I1);
t = cputime;

for j=1:ITERATIONS,
  pedro = features_pedro(I, 8);
end
time1 = cputime-t;

% FLOATING POINT TESTS
I = im2single(I1);

t = cputime;
for j=1:ITERATIONS,
  pedro_float = features_pedro_float(I, 8);
end
time2 = cputime-t;


t = cputime;
for j=1:ITERATIONS,
  mmex = features_madmex(I, 8);
end
time3 = cputime-t;

disp(sprintf('Sequential solution (DOUBLE) %f', time1));
disp(sprintf('Sequential solution %f', time2));
disp(sprintf('Madmex parallel solution %f with speedup %0.4fx, ', time3, time2/time3));

err_pedro = 0;
err_pedro_float = 0;
len_arr = 0;

for i = 1:size(pedro,1)
  for j = 1:size(pedro,2)
    len_arr = len_arr + 1;
    if abs(pedro_float(i,j) - mmex(i,j)) > .01
      err_pedro_float = err_pedro_float + 1;
    end
    if abs(pedro(i,j) - mmex(i,j)) > .01
      err_pedro = err_pedro + 1;
     %display(sprintf('us: %f ref: %f',mmex(i), pedro(i)));
    end
  end
end


display(sprintf('percent deviation from pedro (DOUBLE): %f%% (%d off)', 100.0*err_pedro/len_arr, err_pedro));
display(sprintf('percent deviation from pedro: %f%% (%d off)', 100.0*err_pedro_float/len_arr, err_pedro_float));


quit();
%disp(o);

