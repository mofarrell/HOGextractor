
function [time, out_mat] = iterate_hog(func, I1,OCTAVES, LPO, type)
out_mat = cell(OCTAVES, LPO);
I2 = I1;
if type == 1
  imconv = @(x) im2single(x);
else
  imconv = @(x) im2double(x);
end
I = imconv(I2);
t = cputime;
for j=1:OCTAVES,
  for i=1:LPO,
    power = 2^(j-1 + (i-1)/LPO);
   % I2 = imresize(I2, 1/power);
   % if type == 1
   %   imconv = @(x) im2single(x);
   % else
   %   imconv = @(x) im2double(x);
   % end
   % I = imconv(I2);
    out_mat{j, i} = func(I, 8);
  end
end
time = cputime-t;
end

