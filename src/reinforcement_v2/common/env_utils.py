import numpy as np
from osim.env.utils.mygym import convert_to_gym

obs_vtgt_space = np.array([[-10] * 2 , [10] * 2 ])

obs_body_space = np.array([[-1.0] * 97, [1.0] * 97])
obs_body_space[:, 0] = [0, 3]  # pelvis height
obs_body_space[:, 1] = [-np.pi, np.pi]  # pelvis pitch
obs_body_space[:, 2] = [-np.pi, np.pi]  # pelvis roll
obs_body_space[:, 3] = [-20, 20]  # pelvis vel (forward)
obs_body_space[:, 4] = [-20, 20]  # pelvis vel (leftward)
obs_body_space[:, 5] = [-20, 20]  # pelvis vel (upward)
obs_body_space[:, 6] = [-10 * np.pi, 10 * np.pi]  # pelvis angular vel (pitch)
obs_body_space[:, 7] = [-10 * np.pi, 10 * np.pi]  # pelvis angular vel (roll)
obs_body_space[:, 8] = [-10 * np.pi, 10 * np.pi]  # pelvis angular vel (yaw)
obs_body_space[:, [9 + x for x in [0, 44]]] = np.array(
    [[-5, 5]]).transpose()  # (r,l) ground reaction force normalized to bodyweight (forward)
obs_body_space[:, [10 + x for x in [0, 44]]] = np.array(
    [[-5, 5]]).transpose()  # (r, l) ground reaction force normalized to bodyweight (rightward)
obs_body_space[:, [11 + x for x in [0, 44]]] = np.array(
    [[-10, 10]]).transpose()  # (r, l) ground reaction force normalized to bodyweight (upward)
obs_body_space[:, [12 + x for x in [0, 44]]] = np.array(
    [[-45 * np.pi / 180, 90 * np.pi / 180]]).transpose()  # (r, l) joint: (+) hip abduction
obs_body_space[:, [13 + x for x in [0, 44]]] = np.array(
    [[-180 * np.pi / 180, 45 * np.pi / 180]]).transpose()  # (r, l) joint: (+) hip extension
obs_body_space[:, [14 + x for x in [0, 44]]] = np.array(
    [[-180 * np.pi / 180, 15 * np.pi / 180]]).transpose()  # (r, l) joint: (+) knee extension
obs_body_space[:, [15 + x for x in [0, 44]]] = np.array(
    [[-45 * np.pi / 180, 90 * np.pi / 180]]).transpose()  # (r, l) joint: (+) ankle extension (plantarflexion)
obs_body_space[:, [16 + x for x in [0, 44]]] = np.array(
    [[-5 * np.pi, 5 * np.pi]]).transpose()  # (r, l) joint: (+) hip abduction
obs_body_space[:, [17 + x for x in [0, 44]]] = np.array(
    [[-5 * np.pi, 5 * np.pi]]).transpose()  # (r, l) joint: (+) hip extension
obs_body_space[:, [18 + x for x in [0, 44]]] = np.array(
    [[-5 * np.pi, 5 * np.pi]]).transpose()  # (r, l) joint: (+) knee extension
obs_body_space[:, [19 + x for x in [0, 44]]] = np.array(
    [[-5 * np.pi, 5 * np.pi]]).transpose()  # (r, l) joint: (+) ankle extension (plantarflexion)
obs_body_space[:, [20 + x for x in list(range(0, 33, 3)) + list(range(44, 77, 3))]] = np.array(
    [[0, 3]]).transpose()  # (r, l) muscle forces, normalized to maximum isometric force
obs_body_space[:, [21 + x for x in list(range(0, 33, 3)) + list(range(44, 77, 3))]] = np.array(
    [[0, 3]]).transpose()  # (r, l) muscle lengths, normalized to optimal length
obs_body_space[:, [22 + x for x in list(range(0, 33, 3)) + list(range(44, 77, 3))]] = np.array(
    [[-50, 50]]).transpose()  # (r, l) muscle velocities, normalized to optimal length per second
observation_space = np.concatenate((obs_vtgt_space, obs_body_space), axis=1)
observation_space = convert_to_gym(observation_space)


def vtgt_field_to_single_vec(field):
    return np.array(field)[:, 5, 5]


def obs_dict_to_list(obs_dict):
    LENGTH0 = 1.

    # Augmented environment from the L2R challenge
    res = []

    # target velocity field (in body frame)
    v_tgt = np.ndarray.flatten(obs_dict['v_tgt_field'])
    res += v_tgt.tolist()

    res.append(obs_dict['pelvis']['height'])
    res.append(obs_dict['pelvis']['pitch'])
    res.append(obs_dict['pelvis']['roll'])
    res.append(obs_dict['pelvis']['vel'][0] / LENGTH0)
    res.append(obs_dict['pelvis']['vel'][1] / LENGTH0)
    res.append(obs_dict['pelvis']['vel'][2] / LENGTH0)
    res.append(obs_dict['pelvis']['vel'][3])
    res.append(obs_dict['pelvis']['vel'][4])
    res.append(obs_dict['pelvis']['vel'][5])

    for leg in ['r_leg', 'l_leg']:
        res += obs_dict[leg]['ground_reaction_forces']
        res.append(obs_dict[leg]['joint']['hip_abd'])
        res.append(obs_dict[leg]['joint']['hip'])
        res.append(obs_dict[leg]['joint']['knee'])
        res.append(obs_dict[leg]['joint']['ankle'])
        res.append(obs_dict[leg]['d_joint']['hip_abd'])
        res.append(obs_dict[leg]['d_joint']['hip'])
        res.append(obs_dict[leg]['d_joint']['knee'])
        res.append(obs_dict[leg]['d_joint']['ankle'])
        for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
            res.append(obs_dict[leg][MUS]['f'])
            res.append(obs_dict[leg][MUS]['l'])
            res.append(obs_dict[leg][MUS]['v'])
    return res


def normalize_obs(obs):
    low = observation_space.low
    high = observation_space.high
    normed_obs = 2 * (obs - low) / (high - low) - 1.
    return normed_obs


def prepare_obs(obs_dict):
    obs_dict['v_tgt_field'] = vtgt_field_to_single_vec(obs_dict['v_tgt_field'])
    obs_list = obs_dict_to_list(obs_dict)
    normed_obs_list = normalize_obs(obs_list)
    return normed_obs_list