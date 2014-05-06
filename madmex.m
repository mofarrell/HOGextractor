I1 = imread('sample2.jpg');

OCTAVES = 10;
LPO = 10;
ITERATIONS = 1;

display(sprintf('==== LOADING STANDARD IMPLEMENTATIONS ====\n'));

% STANDARD DOUBLE

[time1, pedro_double] = iterate_hog(@features_pedro, I1, OCTAVES, LPO, 2);

% STANDARD SINGLE

[time2, pedro_single] = iterate_hog(@features_pedro_single, I1, OCTAVES, LPO, 1);



display(sprintf('==== TESTING LOOKUP TABLE ====\n'));

[time4, lookup] = iterate_hog(@features_lookup, I1, OCTAVES, LPO, 2);


% COMPILATION TARGET

if type == 1
  display(sprintf('==== USING SINGLE ====\n'));
else
  display(sprintf('==== USING DOUBLE ====\n'));
end


display(sprintf('==== TESTING SIMD PARALLELIZED ====\n'));

[time3, mmex] = iterate_hog(@features_madmex, I1, OCTAVES, LPO, type);

display(sprintf('==== TESTING SIMD PARALLELIZED LOOKUP TABLE ====\n'));

[time5, lookup_mmex] = iterate_hog(@features_lookup_madmex, I1, OCTAVES, LPO, type);


display(sprintf('\n----------------------- TIMING RESULTS -----------------------\n'));
disp(sprintf('Sequential solution (DOUBLE) %f', time1));
disp(sprintf('Sequential solution %f', time2));
disp(sprintf('Madmex parallel solution %f with speedup %0.4fx over standard implementation ', time3, time1/time3));
disp(sprintf('Madmex lookup solution %f with speedup %0.4fx over standard implementation', time4, time1/time4));
disp(sprintf('Madmex lookup parallel solution %f with speedup %0.4fx over standard implementation', time5, time1/time5));


% % % % % % % % % % %
% CORRECTNESS TEST  %
% % % % % % % % % % % 

%err_mmex_double = 0;
%err_mmex_single = 0;
%len_arr = 0;
%
%err_lookup_double = 0;
%err_lookup_single = 0;
%
%err_lookup_parallel_single = 0;
%err_lookup_parallel_double = 0;
%
%for i = 1:size(pedro_double,1)
%  for j = 1:size(pedro_double,2)
%    len_arr = len_arr + 1;
%    if abs(pedro_single(i,j) - mmex(i,j)) > .01
%      err_mmex_single = err_mmex_single + 1;
%    end
%    if abs(pedro_double(i,j) - mmex(i,j)) > .01
%      err_mmex_double = err_mmex_double + 1;
%    end
%  end
%end
%
%for i = 1:size(pedro_double,1)
%  for j = 1:size(pedro_double,2)
%    if abs(pedro_double(i,j) - lookup(i,j)) > .01
%      err_lookup_double = err_lookup_double + 1;
%    end
%  end
%end
%
%for i = 1:size(pedro_single,1)
%  for j = 1:size(pedro_single,2)
%    if abs(pedro_single(i,j) - lookup_mmex(i,j)) > .01
%      err_lookup_parallel_single = err_lookup_parallel_single + 1;
%    end
%    if abs(pedro_double(i,j) - lookup_mmex(i,j)) > .01
%      err_lookup_parallel_double = err_lookup_parallel_double + 1;
%    end
%  end
%end
%
%display(sprintf('\n-------------------- CORRECTNESS RESULTS ---------------------\n'));
%
%display(sprintf('--- COMPARISONS TO DOUBLE ---\n'))
%display(sprintf('Parallel (no lookup) percent deviation: %f%% (%d off)', 100.0*err_mmex_double/len_arr, err_mmex_double));
%display(sprintf('Lookup (no parallel) percent deviation: %f%% (%d off)', 100.0*err_lookup_double/len_arr, err_lookup_double));
%display(sprintf('Lookup Parallel percent deviation: %f%% (%d off)', 100.0*err_lookup_parallel_double/len_arr, err_lookup_parallel_double));
%
%display(sprintf('\n--- COMPARISONS TO SINGLE ---\n'))
%display(sprintf('Parallel (no lookup) percent deviation: %f%% (%d off)', 100.0*err_mmex_single/len_arr, err_mmex_single));
%display(sprintf('Lookup Parallel percent deviation: %f%% (%d off)', 100.0*err_lookup_parallel_single/len_arr, err_lookup_parallel_single));


quit();

