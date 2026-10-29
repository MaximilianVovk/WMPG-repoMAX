#!/usr/bin/env python3
"""
Convert an EN fireball network solution (.res/.inp/.lc, Boltrack output) into a WMPL trajectory pickle,
shaped as close as possible to a pickle produced by the WMPL trajectory solver
(wmpl.Trajectory.Trajectory with wmpl.Trajectory.Trajectory.ObservedPoints and a wmpl.Trajectory.Orbit.Orbit).

What comes from the EN files:
    - .inp: for each camera the time (s), length along the trajectory (km) and height (km) of every measured
      point, the ECI (of date) position of the trajectory beginning (km), the apparent radiant unit vector and
      the Greenwich sidereal time (rad).
    - .lc:  calibrated absolute magnitudes (with their sd) per station, mapped on the astrometric points of
      the camera the light curve was measured on (interpolated in length, never extrapolated).
    - .res: Julian date, begin/end points, apparent radiant, velocities and the camera list.
The orbit is recomputed with wmpl.Trajectory.Orbit.calcOrbit and the per-camera length, lag, velocities
and initial velocity are computed as wmpl.Trajectory.Trajectory does.

The EN files do not have the station coordinates nor the lines of sight, so all the station/angular fields
(lat, lon, ele, ra/dec, azim/elev, meas_range, residuals, ...) are left as None. As there is no range, the
apparent magnitudes are not known and obs.magnitudes is set to the absolute magnitudes.
Cameras without photometry have NaN absolute magnitudes.

EN photometry is calibrated with a zero magnitude power of 1500 W (the radiated energy in the .res is
recovered from the light curve with it), this is stored in traj.P_0m.

Usage:
    python en_fireball_to_wmpl_pickle.py path/to/EN040326_201155
"""

import argparse
import collections
import glob
import os
import re

import numpy as np
import scipy.optimize

from wmpl.Trajectory.Trajectory import Trajectory, ObservedPoints
from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Utils.Pickling import savePickle
from wmpl.Utils.TrajConversions import ecef2LatLonAlt, geo2Cartesian


# Zero magnitude power of the EN photometry (W)
EN_P_0M = 1500.0


def lineFunc(x, m, k):
    return m*x + k


def readRes(res_path):
    """ Read the values needed from the Boltrack .res summary file. """

    txt = open(res_path).read()

    def nums(label, n):
        match = re.search(label + r"([^\n]*)", txt)
        return [float(x) for x in re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", match.group(1))][:n]

    res = {}
    res['code'] = re.search(r"METEOR CODE:\s*(\S+)", txt).group(1)
    res['jd'] = nums(r"JULIAN DATE:", 1)[0]
    res['beg'] = nums(r"\n BEGINNING ", 3)
    res['end'] = nums(r"\n END ", 3)
    res['radiant_apparent'] = nums(r"\nAPPARENT ", 3)
    res['v_init_stddev'] = 1000*float(re.search(r"\n\(at beginning\)\s+\S+\s+\S+\s+(\S+)", txt).group(1))
    res['v_avg'] = 1000*nums(r"VELOCITY AT THE AVERAGE TRAJECTORY POINT:", 1)[0]
    res['cameras'] = re.search(r"NUMBER OF CAMERAS:[^\n]*\n\(([^)]*)\)", txt).group(1).split(',')

    return res


def readInp(inp_path):
    """ Read the .inp file, one block per camera: header, ECI beginning point, radiant and the
    (time, length, height) points. The first and last rows (time 0) are the begin/end markers. """

    blocks = []
    lines = [l.split() for l in open(inp_path) if l.strip()]
    i = 0
    while i < len(lines):
        if len(lines[i]) == 4 and lines[i][1] == 'EN':
            block = {'station': lines[i][3]}
            block['beg_eci'] = 1000*np.array([float(x) for x in lines[i + 2]])
            rad_gst = [float(x) for x in lines[i + 3]]
            block['radiant_eci'] = np.array(rad_gst[:3])
            block['gst'] = rad_gst[3]
            rows = []
            i += 5  # skip the header lines and the begin marker
            while float(lines[i][0]) != -1000:
                rows.append([float(x) for x in lines[i]])
                i += 1
            rows = np.array(rows[:-1])  # drop the end marker
            block['time'], block['length'], block['height'] = rows[:, 0], 1000*rows[:, 1], 1000*rows[:, 2]
            blocks.append(block)
        i += 1

    return blocks


def readLc(lc_path):
    """ Read the .lc file: per station (time, abs. magnitude, sd, length, height). """

    lc = {}
    station = None
    for line in open(lc_path):
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            station = parts[0]
            lc[station] = []
        elif len(parts) == 5 and station is not None:
            lc[station].append([float(x) for x in parts])

    return {st: {'time': np.array(r)[:, 0], 'mag': np.array(r)[:, 1], 'mag_sd': np.array(r)[:, 2],
                 'length': 1000*np.array(r)[:, 3]} for st, r in lc.items() if r}


def eciToGeo(eci, gst):
    """ ECI (of date) to geodetic coordinates with the EN sidereal time, so the positions match the .res. """

    x = eci[:, 0]*np.cos(gst) + eci[:, 1]*np.sin(gst)
    y = -eci[:, 0]*np.sin(gst) + eci[:, 1]*np.cos(gst)
    lat, lon, ele = np.array([ecef2LatLonAlt(*p) for p in np.c_[x, y, eci[:, 2]]]).T

    return lat, lon, ele


def computeVelocities(obs, v_init_part):
    """ Velocities and lag as in wmpl Trajectory.calcVelocity and calcLag. """

    # Average velocity including all points up to the given point (first 4 points from the first 4)
    velocities_prev_point = []
    for i in range(len(obs.time_data)):
        n = 4 if i < 4 else i + 1
        popt, _ = scipy.optimize.curve_fit(lineFunc, obs.time_data[:n], obs.length[:n])
        velocities_prev_point.append(popt[0])
    obs.velocities_prev_point = np.array(velocities_prev_point)

    # Point to point velocities (first is always 0)
    time_diffs = obs.time_data - np.r_[0, obs.time_data][:-1]
    time_diffs[time_diffs == 0] = np.finfo(np.float64).eps
    obs.velocities = (obs.length - np.r_[0, obs.length][:-1])/time_diffs

    # Lag from a line fitted to the first part of the points
    init_part_size = int(v_init_part*len(obs.time_data))
    if init_part_size < 4:
        init_part_size = len(obs.time_data)
    obs.lag_line, _ = scipy.optimize.curve_fit(lineFunc, obs.time_data[:init_part_size],
                                               obs.length[:init_part_size])
    obs.lag = obs.length - lineFunc(obs.time_data, *obs.lag_line)
    obs.v_init = obs.lag_line[0]


def makeObservation(obs_id, station_id, block, jdt_ref, t_shift, lc_station):
    """ Create an ObservedPoints with the same attributes of a WMPL solved one. """

    obs = ObservedPoints.__new__(ObservedPoints)
    n = len(block['time'])

    # Station and line of sight data, not in the EN files
    for name in ['lat', 'lon', 'ele', 'x_stat', 'y_stat', 'z_stat', 'stat_eci', 'stat_eci_los', 'plane_N',
                 'meas1', 'meas2', 'ra_data', 'dec_data', 'ra_data_los', 'dec_data_los', 'azim_data',
                 'elev_data', 'meas_eci', 'meas_eci_los', 'x_eci', 'y_eci', 'z_eci', 'x_eci_los', 'y_eci_los',
                 'z_eci_los', 'meas_range', 'model_range', 'model_ra', 'model_dec', 'model_azim',
                 'model_elev', 'model_fit1', 'model_fit2', 'ang_res', 'ang_res_std', 'h_residuals',
                 'h_res_rms', 'v_residuals', 'v_res_rms', 'incident_angle', 'v_init_stddev', 'fov_beg',
                 'fov_end', 'excluded_time']:
        setattr(obs, name, None)

    obs.station_id = station_id
    obs.obs_id = obs_id
    obs.comment = "Converted from EN .inp/.lc/.res (Boltrack); no station coordinates nor lines of sight."
    obs.jdt_ref = jdt_ref
    obs.kmeas = n
    obs.ignore_list = np.zeros(n, dtype=np.uint8)
    obs.ignore_station = False
    obs.excluded_indx_range = []
    obs.weight = 1.0
    obs.jacchia_fit = np.zeros(2)

    obs.time_data = block['time'] - t_shift
    obs.JD_data = jdt_ref + obs.time_data/86400.0

    # Positions on the trajectory line from the beginning point (state vector)
    eci = block['beg_eci'] - np.outer(block['length'], block['radiant_eci'])
    lat, lon, _ = eciToGeo(eci, block['gst'])
    obs.model_lat = obs.meas_lat = lat
    obs.model_lon = obs.meas_lon = lon
    obs.model_ht = obs.meas_ht = block['height']
    obs.model_eci = np.array([geo2Cartesian(la, lo, h, jd) for la, lo, h, jd in
                              zip(lat, lon, block['height'], obs.JD_data)])
    obs.state_vect_dist = block['length']
    obs.length = block['length'] - block['length'][0]

    obs.rbeg_lat, obs.rbeg_lon, obs.rbeg_ele, obs.rbeg_jd = lat[0], lon[0], block['height'][0], obs.JD_data[0]
    obs.rend_lat, obs.rend_lon, obs.rend_ele, obs.rend_jd = lat[-1], lon[-1], block['height'][-1], obs.JD_data[-1]
    obs.htmin_lat, obs.htmin_lon, obs.htmin_ele, obs.htmin_jd = obs.rend_lat, obs.rend_lon, obs.rend_ele, obs.rend_jd

    # Photometry measured by this camera, interpolated in length on the astrometric points
    if lc_station is not None:
        inside = (block['length'] >= lc_station['length'].min()) & (block['length'] <= lc_station['length'].max())
        obs.absolute_magnitudes = np.where(inside, np.interp(block['length'], lc_station['length'],
                                                             lc_station['mag']), np.nan)
    else:
        obs.absolute_magnitudes = np.full(n, np.nan)
    # no range to the station, the apparent magnitudes are not known
    obs.magnitudes = obs.absolute_magnitudes.copy()

    return obs


def convertENFolder(folder):

    res_path, = glob.glob(os.path.join(folder, '*.res'))
    base = os.path.splitext(res_path)[0]
    res = readRes(res_path)
    blocks = readInp(base + '.inp')
    lc = readLc(base + '.lc')

    # Name the cameras from the .res list in order (e.g. the 2 blocks of station 102 are 102-30 and 102-72)
    cameras_left = list(res['cameras'])
    for block in blocks:
        camera = next(c for c in cameras_left if c.split('-')[0] == block['station'])
        cameras_left.remove(camera)
        block['camera'] = camera

    # The .lc of a station was measured on its first camera
    lc_camera = {}
    for block in blocks:
        lc_camera.setdefault(block['station'], block['camera'])

    # The .inp times are relative to the .res JD, the reference time is the first point as in WMPL
    t_shift = min(block['time'][0] for block in blocks)
    jdt_ref = res['jd'] + t_shift/86400.0

    v_init = 1000*res['radiant_apparent'][2]
    traj = Trajectory(jdt_ref, output_dir=folder, meastype=1, v_init_part=0.25, monte_carlo=False,
                      show_plots=False, save_results=False, verbose=False)

    traj.observations = []
    for i, block in enumerate(blocks):
        lc_station = lc.get(block['station']) if lc_camera[block['station']] == block['camera'] else None
        obs = makeObservation(i + 1, 'cz_' + block['camera'], block, jdt_ref, t_shift, lc_station)
        computeVelocities(obs, traj.v_init_part)
        traj.observations.append(obs)

    # Beginning point (state vector) and radiant
    radiant_eci = blocks[0]['radiant_eci']
    beg_lat, beg_lon, beg_ele = [x[0] for x in eciToGeo(blocks[0]['beg_eci'][None, :], blocks[0]['gst'])]
    beg_ele = 1000*res['beg'][2]
    t_beg = min(obs.time_data[0] - obs.state_vect_dist[0]/v_init for obs in traj.observations)
    rbeg_jd = jdt_ref + t_beg/86400.0
    state_vect = np.array(geo2Cartesian(beg_lat, beg_lon, beg_ele, rbeg_jd))

    end_lat, end_lon = np.radians(res['end'][1]), np.radians(res['end'][0])
    end_ele = 1000*res['end'][2]
    rend_jd = max(obs.JD_data[-1] for obs in traj.observations)

    traj.file_name = res['code']
    traj.traj_id = res['code']
    traj.P_0m = EN_P_0M
    traj.state_vect = traj.state_vect_mini = state_vect
    traj.state_vect_avg = np.mean([obs.model_eci.mean(axis=0) for obs in traj.observations], axis=0)
    traj.radiant_eci_mini = traj.avg_radiant = radiant_eci
    traj.radiant_eq = traj.radiant_eq_mini = (np.radians(res['radiant_apparent'][0]),
                                             np.radians(res['radiant_apparent'][1]))
    traj.v_init = v_init
    traj.v_init_stddev = res['v_init_stddev']
    traj.v_avg = res['v_avg']
    traj.velocity_fit = np.array([v_init, 0.0])
    traj.rbeg_lat, traj.rbeg_lon, traj.rbeg_ele, traj.rbeg_jd = beg_lat, beg_lon, beg_ele, rbeg_jd
    traj.rend_lat, traj.rend_lon, traj.rend_ele, traj.rend_jd = end_lat, end_lon, end_ele, rend_jd
    traj.htmin_lat, traj.htmin_lon, traj.htmin_ele, traj.htmin_jd = end_lat, end_lon, end_ele, rend_jd
    traj.rbeg_ele_wgs84, traj.rend_ele_wgs84, traj.htmin_ele_wgs84 = beg_ele, end_ele, end_ele
    traj.jd_avg = res['jd']
    traj.meas_count = sum(len(obs.time_data) for obs in traj.observations)
    traj.stations_time_dict = collections.OrderedDict((obs.station_id, True) for obs in traj.observations)
    traj.t_ref_station = 0
    traj.time_diffs = traj.time_diffs_final = np.zeros(len(traj.observations))
    traj.jacchia_fit = np.zeros(2)
    traj.intersection_list = []
    traj.best_conv_inter = None
    traj.incident_angles = []

    traj.orbit = calcOrbit(radiant_eci, v_init, traj.v_avg, state_vect, rbeg_jd, stations_fixed=True,
                           reference_init=True, rotation_correction=True, v_init_stddev_direct=traj.v_init_stddev)
    traj.v0z = -v_init*np.cos(traj.orbit.zc)

    file_name = res['code'] + '_trajectory.pickle'
    savePickle(traj, folder, file_name)
    print('Saved:', os.path.join(folder, file_name))

    return traj


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Convert an EN fireball (.res/.inp/.lc) into a WMPL trajectory pickle.")
    arg_parser.add_argument('folder', type=str, help="Folder with the EN .res, .inp and .lc files.")
    cml_args = arg_parser.parse_args()

    traj = convertENFolder(cml_args.folder)

    orb = traj.orbit
    print(f"Orbit: a={orb.a:.5f} AU e={orb.e:.5f} q={orb.q:.5f} AU i={np.degrees(orb.i):.4f} "
          f"peri={np.degrees(orb.peri):.4f} node={np.degrees(orb.node):.4f} v_g={orb.v_g/1000:.4f} km/s")
    for obs in traj.observations:
        print(f"{obs.station_id}: {len(obs.time_data)} points, v_init={obs.v_init:.1f} m/s, "
              f"photometric points={np.sum(np.isfinite(obs.absolute_magnitudes))}")
