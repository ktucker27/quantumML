function yp = gdtwa_eom(~, y, hamu, hamw, npart, nlevels)

d = nlevels*nlevels;

for part_idx1 = 1:npart
    start_idx1 = (part_idx1-1)*d + 1;
    
    sum1 = zeros(d, 1); % First order term
    sum2 = zeros(d, 1); % Second order term
    for mu = 1:d
        for nu = 1:d
            [c, idx] = gm_comm(mu, nu, nlevels);
            for ii=1:size(idx,1)
                sum1(mu, 1) = sum1(mu, 1) + c(ii)*hamu(start_idx1 - 1 + mu)*y(start_idx1 - 1 + idx(ii));
            end
            
            for part_idx2 = 1:npart
                if part_idx2 == part_idx1
                    continue
                end
                
                start_idx2 = (part_idx2-1)*d + 1;
                
                for sig = 1:d
                    [c, idx] = gm_comm(mu, sig, nlevels);
                    for ii=1:size(idx,1)
                        sum2(mu, 1) = sum2(mu, 1) + c(ii)*hamw(part_idx1, mu, part_idx2, nu)*y(start_idx1 - 1 + idx(ii))*y(start_idx2 - 1 + nu);
                    end
                end
            end
        end
    end
    
    yp(start_idx1:start_idx1 + d - 1, 1) = -1i*(sum1 + sum2);
end