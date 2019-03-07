function ev = nqs_ev(oloc, wave, a, b, w, num_samps, num_steps)
% nqs_ev Sample the distribution given by the wave function corresponding
%        to the parameters (a,b,w) num_samp times to estimate the value of
%        the local operator given by the function oloc(a, b, w, sz)
%
%        oloc is a function handle or name that perorms the local operator
%        calculation by exploiting the sparsity of the operator to avoid
%        any exponentially sized sums

ev = 0;
for i=1:num_samps
    sz = nqs_sample(wave, a, b, w, num_steps);
    ev = ev + feval(oloc, wave, a, b, w, sz);
end

ev = ev/num_samps;