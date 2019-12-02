## 2/12/2019
 - Add simple reward function based on DTW-distance:
  $$r = \exp(4. - d(x,y))$$
 - Run ASAC for VTL model
### TODO:
 - In some cases DTW distance is NaN, need to consider this case
 - Check `env.action_space` (we normalize actions to be within (-1;1) range. So unnormalized actions should have sense (i.e. not too big, nor too small))
 - Consider reward function based on the DTW-distance:
 $$r(x,y) = f(d(x,y))$$
