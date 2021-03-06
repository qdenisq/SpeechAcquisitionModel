## 2/12/2019
 - Add simple reward function based on DTW-distance:
  $$r = \exp(4. - d(x,y))$$
 - Run ASAC for VTL model

## 4/12/2019
  - Add env with reference in its obs space. User can select what reference obs consists of by selecting parameters in `selected_reference_state` in the config file.
  _Note_: If want to incplude acoustics of the reference in the obs, include __"ACOUSTICS"__ in the `selected_reference_state`

## 6/12/2019
  - Add backprop into policy algorithm. It still updates policy with l1 distance between predicted state and reference (consider using dtw here somehow)

## 14/04/2020
  - IMPORTANT: change dtwalign plor_path by `return _, ax`
  - IMPORTANT: subproc_vec_env comment `if done:` in `if cmd == 'step':`

### TODO:
 - In some cases DTW distance is NaN, need to consider this case
 - Check `env.action_space` (we normalize actions to be within (-1;1) range. So unnormalized actions should have sense (i.e. not too big, nor too small))
 - Consider reward function based on the DTW-distance:
 $$r(x,y) = f(d(x,y))$$
 - Check noise and how big it is compared to policy actions
 - ~~Somehow let the agent observe the reference (now it receives only its own vocal tract state and sound)~~
 - What is appropriate audio bound (in `base_env`)
 - test new env in `reference_masked_stw_we_env`
 - test backprop into policy
 
