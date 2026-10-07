"""Match DRAMA/SARA ablating components to optical re-entry detections.

Usage:
    python sara_vs_obs.py <SARA_DIR> <REPORT or DIR> [<REPORT or DIR> ...] [--out DIR] [--curved]

    python sara_vs_obs.py 20240710/DRAMA-SARA_V2mini 20240710/skyfit_traj

Directories are searched recursively for *_report.txt (the *_curved_report.txt
files are used instead with --curved).

Steps
 1. SARA: merge every component's AeroThermalHistory across compound phases (by
    UUID), attach lat/lon/downrange from the matching Trajectory file, and find
    where it loses mass.
 2. Observations: read the 'Points' table (lat, lon, height, AbsMag) of each report.
 3. Match by altitude: a component is "recorded" if it loses mass inside the
    altitude range of an observed segment  -> <RunID>.recorded_AltitudeVsDownrange.png
 4. Place every ablating component on the observed track: inside the observed
    altitudes interpolate the observations; outside, move along the observed
    great circle by the SARA downrange difference     -> <RunID>.recorded_GroundMap.png
 5. Light-curve check: altitude vs downrange of every SARA component with the observations
    placed by their along-track distance from the anchor (see below); AbsMag compared with
    the SARA mass-loss rate
                                                      -> <RunID>.lightcurve_check.png
    Speed check: observed point-to-point speed (scatter) vs the SARA speed of the components
    losing mass in each segment, against altitude      -> <RunID>.velocity_check.png
    Anchor: one point ties SARA to the sky - observed segment, altitude, lat/lon and direction
    of travel (local fit of the track), and the reference SARA trajectory's time and downrange
    at that altitude. Chosen among the segments SARA reproduces: the lowest one with a best
    convergence angle >= --anchor_min_qconv and >= 2 stations (or --anchor_segment). Below it
    SARA is placed along the anchor's great circle by downrange relative to the anchor;
    observations below it are an independent check  -> <RunID>.anchor_candidates.csv
 6. Dark flight of the SARA survivors with OpenDarkflight. Start: end of mass loss, or
    --v_start (4.5 km/s) for pieces that never lose mass, placed on the observed track
    extended from the anchor                          -> <RunID>.darkflight_inputs.csv
 7. Strewn field: nominal + Monte Carlo impact points and their convex hull
    -> <RunID>.strewnfield.txt (readable), .strewnfield.kml, .strewnfield_impacts.csv,
       .strewnfield_map.png
    Winds: --winds_file <profile> [--winds_type wrf|wyoming], or --winds_file auto
    [--winds_type auto:gfs|auto:era5|...]; without it the atmosphere is calm.
"""
import argparse
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # map is drawn without coastlines
    ccrs = None

R_EARTH = 6371.0  # km
TRAJ_COLS = ["t", "h", "lat", "lon", "v", "downrange", "drag", "lift", "side",
             "knudsen", "mach", "fpa", "heading"]
AERO_COLS = ["t", "h", "T", "mass", "thick", "q_conv", "q_rad", "q_ox", "q_int", "vis"]


# ----------------------------------------------------------------------------- SARA
def _load(path, cols):
    d = np.loadtxt(path, comments="#", ndmin=2)
    return pd.DataFrame(d[:, :len(cols)], columns=cols[:d.shape[1]])


def load_sara(sara_dir):
    """Return (run_id, components, compound_chain).

    components: {uuid: {'name', 'data' (DataFrame, aero + trajectory), 'area'}}
    compound_chain: trajectory of the parent body (all Compound_of phases).
    """
    sara_dir = Path(sara_dir)
    trajs, chain = {}, []
    for f in sara_dir.glob("*_Trajectory.txt"):
        run_id, rest = f.name.split(".", 1)
        name, key = rest[:-len("_Trajectory.txt")].rsplit(".", 1)
        trajs[key] = _load(f, TRAJ_COLS)
        if name.startswith("Compound_of"):
            chain.append(trajs[key])
    chain = pd.concat(chain).sort_values("t").drop_duplicates("t").reset_index(drop=True)

    segments = defaultdict(list)
    for f in sara_dir.glob("*_AeroThermalHistory.txt"):
        name, tail = f.name.split(".", 1)[1][:-len("_AeroThermalHistory.txt")].rsplit(".", 1)
        compound, _, uuid = tail.rpartition("_")
        traj = trajs[compound or uuid]
        a = _load(f, AERO_COLS)
        if a.empty:
            continue
        for c in ("lat", "downrange", "v", "fpa", "heading"):
            a[c] = np.interp(a.t, traj.t, traj[c])
        a["lon"] = np.interp(a.t, traj.t, np.unwrap(traj.lon, period=360))
        area = re.search(r"ReferenceArea\s*:\s*([\d.eE+-]+)", f.read_text(errors="ignore"))
        segments[uuid].append((a, name, float(area.group(1)) if area else np.nan))

    components = {}
    for uuid, segs in segments.items():
        data = pd.concat([s[0] for s in segs]).sort_values("t").drop_duplicates("t")
        components[uuid] = {"name": segs[0][1], "data": data.reset_index(drop=True),
                            "area": segs[-1][2]}
    return run_id, components, chain


def ablation(data, rel_tol=1e-6):
    """Boolean mask of rows where the component is losing mass (dm < 0 to the next row)."""
    dm = np.diff(data.mass.values, append=data.mass.values[-1])
    return dm < -rel_tol * data.mass.values[0]


# ----------------------------------------------------------------------------- observations
def load_report(path):
    """Points table of a WMPL trajectory report (ignored points dropped)."""
    lines = Path(path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("No,") and "AbsMag" in l)
    header = [c.strip() for c in lines[i].split(",")]
    rows = []
    for l in lines[i + 1:]:
        if not l.strip():
            break
        rows.append([c.strip() for c in l.split(",")])
    df = pd.DataFrame(rows, columns=header).replace("None", np.nan)
    df = df[df["Ignore"] == "0"]
    out = pd.DataFrame({
        "station": df["Station ID"].values,
        "jd": df["JD"].astype(float).values,
        "lat": df["Latitude (deg)"].astype(float).values,
        "lon": df["Longitude (deg)"].astype(float).values,
        "h": df["Height (m)"].astype(float).values / 1000,
        "absmag": df["AbsMag"].astype(float).values,
        # along-track distance from the segment's first point; the report's 'Length' restarts at 0
        # for every station, 'State vect dist' is common to all of them
        "length": (lambda s: s - s.min())(df["State vect dist (m)"].astype(float).values / 1000),
        "v": df["Vel (m/s)"].astype(float).values / 1000,  # point-to-point speed, km/s (noisy)
    })
    stations, ang_err = {}, []
    j = next(k for k, l in enumerate(lines) if l.strip().startswith("ID, Ignored"))
    st_header = [c.strip() for c in lines[j].split(",")]
    for l in lines[j + 1:]:
        if not l.strip():
            break
        c = dict(zip(st_header, (x.strip() for x in l.split(","))))
        stations[c["ID"].split("_")[0]] = (float(c["Lat +N (deg)"]), float(c["Lon +E (deg)"]))
        if c.get("Ignored") == "False":
            ang_err.append(float(c["+/- Obs ang (deg)"]))
    out = out.sort_values("jd").reset_index(drop=True)
    # how well the segment is reconstructed: used to choose the anchor
    q = re.search(r"Qconv\s*=\s*([\d.]+)", "\n".join(lines))
    out.attrs.update(qconv=float(q.group(1)) if q else np.nan,
                     n_stations=out.station.str.split("_").str[0].nunique(),
                     obs_ang_err_deg=float(np.median(ang_err)) if ang_err else np.nan)
    return out, stations


def find_reports(inputs, curved):
    reports = []
    for p in map(Path, inputs):
        if p.is_file():
            reports.append(p)
            continue
        for f in sorted(p.rglob("*_report.txt")):
            if ("_curved_" in f.name) == curved:
                reports.append(f)
    return reports


# ----------------------------------------------------------------------------- geometry
def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360


def gc_distance(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def destination(lat, lon, brg, dist):
    p, d, b = np.radians(lat), np.asarray(dist) / R_EARTH, np.radians(brg)
    p2 = np.arcsin(np.sin(p) * np.cos(d) + np.cos(p) * np.sin(d) * np.cos(b))
    l2 = np.radians(lon) + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p),
                                      np.cos(d) - np.sin(p) * np.sin(p2))
    return np.degrees(p2), np.degrees(l2)


class Anchor:
    """The one point where SARA is tied to the sky; everything below it is placed from here.

    Observed side: segment, altitude h [km], lat/lon [deg], local direction of travel azim [deg]
    and JD, from a local fit of the segment's track at h. Simulated side: the reference SARA
    trajectory (ref, a component or the parent body) and its time t [s], downrange [km] and flight
    path angle at the same altitude.
    """

    def __init__(self, segment, h, lat, lon, azim, jd, ref_uuid, ref_name, ref, quality):
        self.segment, self.h, self.lat, self.lon, self.azim, self.jd = segment, h, lat, lon, azim, jd
        self.ref_uuid, self.ref_name, self.ref, self.quality = ref_uuid, ref_name, ref, quality
        self.t = float(np.interp(h, *_by_height(ref, "t")))
        self.downrange = float(np.interp(h, *_by_height(ref, "downrange")))
        self.fpa = float(np.interp(h, *_by_height(ref, "fpa")))

    def extend(self, downrange):
        """lat, lon of SARA downranges moved along the observed great circle through the anchor."""
        return destination(self.lat, self.lon, self.azim, np.asarray(downrange, float) - self.downrange)

    def downrange_of(self, lat, lon):
        """SARA downrange of observed points: anchor downrange + along-track distance from the anchor."""
        rel = np.radians(bearing(self.lat, self.lon, lat, lon) - self.azim)
        return self.downrange + gc_distance(self.lat, self.lon, lat, lon) * np.cos(rel)

    def direction(self, lat, lon, downrange):
        """Direction of travel [deg] at points of that great circle."""
        b = bearing(lat, lon, self.lat, self.lon)
        az = np.where(np.asarray(downrange) > self.downrange, (b + 180) % 360, b)
        return np.where(gc_distance(lat, lon, self.lat, self.lon) < 1e-3, self.azim, az)

    @property
    def dr_per_km_h(self):
        """SARA downrange shift per km of altitude at the anchor: the cost of an altitude bias."""
        return 1 / np.tan(np.radians(max(abs(self.fpa), 1e-3)))

    def describe(self):
        return (f"{self.segment} at {self.h:.2f} km, lat {self.lat:.5f}, lon {self.lon:.5f}, "
                f"travel azimuth {self.azim:.2f} deg; SARA reference {self.ref_name}: t {self.t:.2f} s, "
                f"downrange {self.downrange:.1f} km, fpa {self.fpa:.2f} deg "
                f"({self.dr_per_km_h:.0f} km downrange per km of altitude)")


def reliable_altitude(obs, h_lo, h_hi):
    """Lowest altitude in [h_lo, h_hi] that at least two stations observe (h_lo if none does).

    The height of a straight-line solution over a curved Earth bottoms out and rises again; near
    that minimum altitude no longer fixes the along-track position. The anchor is kept where the
    fitted descent is still at least half as steep as at the start of the segment.
    """
    a, b, _ = np.polyfit(obs.length, obs.h, 2)
    if a > 0 and b < 0:
        h_lo = max(h_lo, np.polyval([a, b, _], min(-b / (4 * a), obs.length.max())))
    h_hi = max(h_hi, h_lo)
    grid = np.linspace(h_lo, h_hi, 400)
    rng = obs.groupby(obs.station.str.split("_").str[0]).h.agg(["min", "max"])
    n = ((grid[:, None] >= rng["min"].values) & (grid[:, None] <= rng["max"].values)).sum(axis=1)
    return float(grid[n >= 2].min()) if (n >= 2).any() else float(h_lo)


def fit_track_point(obs, h, window_km=30.0):
    """Observed lat, lon, direction of travel and JD at altitude h.

    Local quadratic fits along the track (vs. the report's Length) of the points within
    window_km of h, so a single noisy point does not set the anchor.
    """
    o = obs.sort_values("length")
    lon_ref = o.lon.mean()
    lon = (o.lon - lon_ref + 180) % 360 - 180 + lon_ref
    # first crossing of h by the height of a straight line over a curved Earth (quadratic in length)
    grid = np.linspace(o.length.min(), o.length.max(), 4000)
    hg = np.polyval(np.polyfit(o.length, o.h, 2), grid)
    l_h = grid[np.argmax(hg <= h)] if (hg <= h).any() else grid[np.argmin(hg)]
    near = np.argsort(np.abs(o.length.values - l_h))
    near = near[:max(15, int((np.abs(o.length.values - l_h) <= window_km).sum()))]
    L = o.length.values[near]
    deg = 2 if len(near) >= 6 else 1
    fit = {c: np.polyfit(L - l_h, v[near], deg)
           for c, v in (("lat", o.lat.values), ("lon", lon.values), ("h", o.h.values), ("jd", o.jd.values))}
    ev = lambda c, x: np.polyval(fit[c], x)
    x = np.linspace(L.min(), L.max(), 2000) - l_h
    hx = ev("h", x)
    x0 = x[np.argmax(hx <= h)] if (hx <= h).any() else x[np.argmin(hx)]  # descending crossing
    lat, lon0 = ev("lat", x0), ev("lon", x0)
    azim = bearing(ev("lat", x0 - 0.5), ev("lon", x0 - 0.5), ev("lat", x0 + 0.5), ev("lon", x0 + 0.5))
    return float(lat), float((lon0 + 180) % 360 - 180), float(azim), float(ev("jd", x0))


def choose_anchor(components, recorded, segments, chain, min_qconv=10.0, segment=None):
    """Pick the anchor segment and build the Anchor. Returns (Anchor, candidates DataFrame).

    Candidates are the segments SARA reproduces (a component loses mass inside them). The reference
    trajectory of a segment is its component losing the most mass there, and the anchor altitude is
    the lowest one inside both the segment and that component's mass loss still seen by two
    stations. A segment qualifies when its best convergence angle is >= min_qconv and >= 2 stations
    saw it; of those the lowest anchor wins (shortest extension below it). The lowest segment is
    not used just for being lowest: a poorly constrained one is skipped. `segment` forces a choice.
    Nothing recorded: the parent body is the reference, tied to the lowest observed segment.
    """
    rows = []
    for seg, rec in (recorded.groupby("segment") if len(recorded) else []):
        r = rec.loc[rec.mass_lost_kg.idxmax()]
        obs, q = segments[seg], segments[seg].attrs
        rows.append({"segment": seg, "ref_uuid": r.uuid, "ref_name": r.component,
                     "h_anchor_km": reliable_altitude(obs, r.h_end_km, r.h_start_km),
                     "qconv_deg": q.get("qconv", np.nan), "n_stations": q.get("n_stations", 0),
                     "obs_ang_err_deg": q.get("obs_ang_err_deg", np.nan), "n_points": len(obs),
                     "h_obs_min_km": obs.h.min(), "h_obs_max_km": obs.h.max()})
    cand = pd.DataFrame(rows)
    if len(cand):
        cand["qualifies"] = (cand.qconv_deg >= min_qconv) & (cand.n_stations >= 2)
        if segment is not None:
            if segment not in set(cand.segment):
                raise SystemExit(f"--anchor_segment {segment!r} is not reproduced by SARA; "
                                 f"candidates: {', '.join(cand.segment)}")
            pick = cand[cand.segment == segment].iloc[0]
        elif cand.qualifies.any():
            pick = cand[cand.qualifies].sort_values("h_anchor_km").iloc[0]
        else:
            pick = cand.sort_values("qconv_deg", ascending=False).iloc[0]
            print(f"WARNING: no anchor candidate has Qconv >= {min_qconv} deg and 2 stations; "
                  f"using the best constrained one, {pick.segment} (Qconv {pick.qconv_deg:.1f} deg)")
        cand["chosen"] = cand.segment == pick.segment
        seg, h, ref_uuid, ref_name = pick.segment, pick.h_anchor_km, pick.ref_uuid, pick.ref_name
        ref = components[ref_uuid]["data"]
    else:
        seg = min(segments, key=lambda s: segments[s].h.min())
        h = reliable_altitude(segments[seg], segments[seg].h.min(), segments[seg].h.max())
        ref_uuid, ref_name, ref = None, "parent body (compound)", chain
        print("WARNING: no SARA component loses mass inside an observed segment; anchoring the "
              f"parent body at {seg}")
    lat, lon, azim, jd = fit_track_point(segments[seg], h)
    return Anchor(seg, h, lat, lon, azim, jd, ref_uuid, ref_name, ref, segments[seg].attrs), cand


def anchor_check(anchor, segments):
    """Independent check: the anchored SARA reference vs. the observations past the anchor.

    Every observed point further along the track than the anchor is projected on the anchor's
    great circle (along, cross [km]). SARA's reference altitude at downrange anchor + along is
    compared with the observed one: dh > 0 means the observation is higher than SARA there, and
    dh * (downrange per km of altitude) is the along-track shift an altitude match would make.
    Compared at the same position, not the same altitude, so a flat track stays well conditioned.
    """
    rows = []
    o_dr = np.argsort(anchor.ref.downrange.values)
    ref_dr, ref_h = anchor.ref.downrange.values[o_dr], anchor.ref.h.values[o_dr]
    for seg, obs in segments.items():
        dist = gc_distance(anchor.lat, anchor.lon, obs.lat.values, obs.lon.values)
        rel = np.radians(bearing(anchor.lat, anchor.lon, obs.lat.values, obs.lon.values) - anchor.azim)
        along, cross = dist * np.cos(rel), dist * np.sin(rel)
        past = (along > 0.5) & (anchor.downrange + along <= ref_dr.max())
        if not past.any():
            continue
        dh = obs.h.values[past] - np.interp(anchor.downrange + along[past], ref_dr, ref_h)
        rows.append({"segment": seg, "n_points": int(past.sum()), "along_max_km": along[past].max(),
                     "dh_median_km": np.median(dh), "dh_at_farthest_km": dh[np.argmax(along[past])],
                     "cross_median_km": np.median(cross[past]),
                     "equiv_shift_km": np.median(dh) * anchor.dr_per_km_h})
    return pd.DataFrame(rows)


class ObservedTrack:
    """Maps a SARA (altitude, downrange) onto the observed ground track.

    Below the anchor altitude: the anchor's great circle, by SARA downrange relative to the anchor.
    Between the anchor and the top of the observations: interpolation of the observed points.
    Above the top: back along the observed track from the highest point.
    """

    def __init__(self, obs, chain, anchor):
        self.anchor = anchor
        o = obs.sort_values("jd")
        self.lon_ref = o.lon.mean()
        lon = (o.lon.values - self.lon_ref + 180) % 360 - 180 + self.lon_ref
        # heights of a straight-line fit can rise again near the end; force monotonic
        h = np.minimum.accumulate(o.h.values)
        keep = np.r_[True, np.diff(h) < 0]
        self.h, self.lat, self.lon = h[keep][::-1], o.lat.values[keep][::-1], lon[keep][::-1]
        self.first = (self.lat[-1], self.lon[-1])  # highest point
        self.back_first = bearing(*self.first, self.lat[0], self.lon[0]) + 180  # opposite to flight at the top
        # SARA downrange of the parent body at the observed top altitude
        ch = chain.sort_values("h")
        self.d_top = np.interp(self.h.max(), ch.h, ch.downrange)

    def place(self, h, downrange):
        """lat, lon, extrapolated? for SARA points (altitude km, downrange km)."""
        h, dr = np.atleast_1d(h).astype(float), np.atleast_1d(downrange).astype(float)
        lat, lon = np.interp(h, self.h, self.lat), np.interp(h, self.h, self.lon)
        lo, hi = h < self.anchor.h, h > self.h.max()
        lat[lo], lon[lo] = self.anchor.extend(dr[lo])
        lat[hi], lon[hi] = destination(*self.first, self.back_first, self.d_top - dr[hi])
        return lat, lon, lo | hi


# ----------------------------------------------------------------------------- analysis
def group_name(name):
    """'RF_numerical_group_3_-_Al_case' -> 'RF numerical group N - Al case'."""
    return re.sub(r"_(\d+)_", "_N_", name).replace("_", " ")


def match(components, segments):
    """One row per (ablating component, observed segment it overlaps with)."""
    rows = []
    for uuid, c in components.items():
        d = c["data"]
        abl = ablation(d)
        if not abl.any():
            continue
        for seg, obs in segments.items():
            lo, hi = obs.h.min(), obs.h.max()
            inside = abl & (d.h >= lo) & (d.h <= hi)
            if not inside.any():
                continue
            o = obs[(obs.h >= d.h[inside].min()) & (obs.h <= d.h[inside].max())]
            rows.append({
                "segment": seg, "component": c["name"], "group": group_name(c["name"]), "uuid": uuid,
                "h_start_km": d.h[inside].max(), "h_end_km": d.h[inside].min(),
                "mass_lost_kg": d.mass[inside].max() - d.mass[inside].min() if inside.sum() > 1 else 0.0,
                "obs_points": len(o), "absmag_min": o.absmag.min(), "absmag_mean": o.absmag.mean(),
                "obs_lat_start": o.lat.iloc[0] if len(o) else np.nan,
                "obs_lon_start": o.lon.iloc[0] if len(o) else np.nan,
                "obs_lat_end": o.lat.iloc[-1] if len(o) else np.nan,
                "obs_lon_end": o.lon.iloc[-1] if len(o) else np.nan,
            })
    return pd.DataFrame(rows)


def group_colors(groups):
    cmap = plt.get_cmap("tab20")
    return {g: cmap(i % 20) for i, g in enumerate(sorted(groups))}


def plot_altitude_downrange(components, segments, recorded, chain, track, out):
    rec_groups = set(recorded.group) if len(recorded) else set()
    colors = group_colors(rec_groups)
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(15, 8), sharey=True,
                                  gridspec_kw={"width_ratios": [3, 1]})
    ax.plot(chain.downrange, chain.h, color="k", lw=1.5, label="parent body (compound)")
    seen = set()
    for c in components.values():
        d, abl = c["data"], ablation(c["data"])
        ax.plot(d.downrange, d.h, color="0.8", lw=0.6, zorder=1)
        if not abl.any():
            continue
        g = group_name(c["name"])
        col = colors.get(g, "0.45")
        lab = g if g in rec_groups and g not in seen else None
        seen.add(g)
        ax.plot(d.downrange.where(abl), d.h.where(abl), color=col, lw=2.5 if lab or g in rec_groups else 1.2,
                label=lab, zorder=3 if g in rec_groups else 2)

    # observed points at the anchor's SARA downrange plus their along-track distance from the
    # anchor: points below the anchor are an independent check of the SARA altitude-downrange
    an = track.anchor
    for k, (seg, obs) in enumerate(segments.items()):
        col = plt.get_cmap("Set1")(k)
        for a in (ax, axm):
            a.axhspan(obs.h.min(), obs.h.max(), color=col, alpha=0.15, zorder=0)
        ax.text(0.005, obs.h.max(), seg, color=col, va="bottom", fontsize=9, transform=ax.get_yaxis_transform())
        ax.scatter(an.downrange_of(obs.lat.values, obs.lon.values), obs.h, s=4, color=col, zorder=4)
        axm.scatter(obs.absmag, obs.h, s=6, color=col, label=seg)
    ax.plot(an.downrange, an.h, "*", color="tab:blue", mec="k", ms=14, zorder=6,
            label=f"anchor ({an.segment}, {an.h:.2f} km)")

    # SARA total mass-loss per km of altitude, for comparison with brightness
    bins = np.arange(0, chain.h.max() + 1, 1.0)
    loss = np.zeros(len(bins) - 1)
    for c in components.values():
        d = c["data"]
        dm = -np.diff(d.mass.values)
        np.add.at(loss, np.clip(np.digitize(d.h.values[:-1], bins) - 1, 0, len(loss) - 1), np.clip(dm, 0, None))
    axl = axm.twiny()
    axl.barh(bins[:-1] + 0.5, loss, height=1.0, color="0.5", alpha=0.35)
    axl.set_xlabel("SARA mass lost per km altitude [kg]", color="0.4")

    ax.set_xlabel("SARA downrange [km]  (observations: anchor downrange + along-track distance from the anchor)")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("SARA components losing mass vs. recorded altitudes (coloured = recorded)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    axm.invert_xaxis()
    axm.set_xlabel("Absolute magnitude")
    axm.legend(fontsize=8, loc="lower left")
    axm.grid(alpha=0.3)
    lo = min(o.h.min() for o in segments.values())
    ylim = (max(0, lo - 30), max(o.h.max() for o in segments.values()) + 10)
    ax.set_ylim(ylim)
    dr = np.concatenate([c["data"].downrange[c["data"].h.between(*ylim)] for c in components.values()]
                        + [chain.downrange[chain.h.between(*ylim)]])
    ax.set_xlim(dr.min() - 50, dr.max() + 50)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def scale_bar(ax, frac=0.2, n_seg=4):
    """Black-and-white map scale bar in the lower-right corner, about `frac` of the map width."""
    from matplotlib.patches import Rectangle
    x0, x1, y0, y1 = ax.get_extent() if ccrs else (*ax.get_xlim(), *ax.get_ylim())
    lat = y0 + 0.05 * (y1 - y0)
    width_km = np.radians(x1 - x0) * R_EARTH * np.cos(np.radians(lat))
    mag = 10 ** np.floor(np.log10(frac * width_km))
    length = max(m * mag for m in (1, 2, 5) if m * mag <= frac * width_km)
    w = length / width_km  # bar width in axes fraction
    left, bottom, h = 0.95 - w, 0.04, 0.012
    ax.add_patch(Rectangle((left - 0.015, bottom - 0.015), w + 0.03, h + 0.06, transform=ax.transAxes,
                           fc="white", ec="0.5", lw=0.5, alpha=0.85, zorder=20))
    for i in range(n_seg):
        ax.add_patch(Rectangle((left + i * w / n_seg, bottom), w / n_seg, h, transform=ax.transAxes,
                               fc="k" if i % 2 == 0 else "white", ec="k", lw=0.8, zorder=21))
    label = f"{length * 1e3:g} m" if length < 1 else f"{length:g} km"
    ax.text(left + w / 2, bottom + h + 0.006, label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, zorder=22)

def plot_ground_map(components, segments, recorded, stations, chain, track, impacts, survivors, out):
    rec_groups = set(recorded.group) if len(recorded) else set()
    colors = group_colors(rec_groups)
    fig = plt.figure(figsize=(14, 9))
    if ccrs:
        ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=track.lon_ref))
        tr = {"transform": ccrs.PlateCarree()}
        ax.add_feature(cfeature.LAND, color="0.93")
        ax.add_feature(cfeature.COASTLINE, lw=0.6)
        ax.add_feature(cfeature.BORDERS, lw=0.3)
        ax.gridlines(draw_labels=True, lw=0.3)
    else:
        ax, tr = fig.add_subplot(), {}
        ax.grid(alpha=0.3)
    unwrap = lambda lon: (np.asarray(lon) - track.lon_ref + 180) % 360 - 180 + track.lon_ref

    ax.plot(unwrap(chain.lon), chain.lat, "--", color="0.5", lw=1, label="SARA ground track (as simulated)", **tr)

    # predicted mass loss placed on the observed track
    seen, all_lat, all_lon = set(), [], []
    for c in components.values():
        d, abl = c["data"], ablation(c["data"])
        if not abl.any():
            continue
        lat, lon, _ = track.place(d.h[abl], d.downrange[abl])
        g = group_name(c["name"])
        lab = g if g not in seen else None
        seen.add(g)
        ax.plot(lon, lat, ".", ms=2, color=colors.get(g, "0.4"), label=lab if g in rec_groups else None, **tr)
        all_lat += list(lat)
        all_lon += list(lon)
    ax.plot([], [], ".", color="0.4", label="other ablating components (not recorded)")

    # what was actually seen
    obs = pd.concat(segments.values())
    sc = ax.scatter(unwrap(obs.lon), obs.lat, c=obs.absmag, cmap="inferno_r", s=10, zorder=5,
                    label="observed (colour = AbsMag)", **tr)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Absolute magnitude")
    for s, (la, lo) in stations.items():
        ax.plot(unwrap(lo), la, "^", color="tab:blue", ms=6, **tr)
        ax.text(unwrap(lo), la, f" {s}", fontsize=7, color="tab:blue", **tr)
    ax.plot([], [], "^", color="tab:blue", label="stations")
    an = track.anchor
    ax.plot(an.lon, an.lat, "*", color="tab:blue", mec="k", ms=14, zorder=7,
            label=f"anchor ({an.segment}, {an.h:.2f} km)", **tr)

    if len(impacts):
        ax.plot(unwrap(impacts.lon), impacts.lat, "x", color="0.4", label="SARA impact points (as simulated)", **tr)
    if len(survivors):
        ax.plot(unwrap(survivors.lon), survivors.lat, "*", color="red", ms=10, zorder=6,
                label="dark-flight start points of the survivors", **tr)

    lat_all = np.r_[obs.lat, all_lat, impacts.lat]  # SARA impacts show the offset of the simulation
    lon_all = unwrap(np.r_[obs.lon, all_lon, impacts.lon])
    ext = [lon_all.min() - 3, lon_all.max() + 3, lat_all.min() - 3, lat_all.max() + 3]
    if ccrs:
        ax.set_extent(ext, crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
        ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    scale_bar(ax)
    ax.set_title("Observed vs. predicted mass loss (SARA ablation placed on the observed track)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def load_impacts(sara_dir):
    rows = []
    for f in Path(sara_dir).glob("*.ImpactingFragments.xml"):
        txt = f.read_text()
        for frag in re.findall(r"<fragment>(.*?)</fragment>", txt, re.S):
            get = lambda tag: re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", frag, re.S).group(1).strip()
            rows.append({"name": get("name"), "uuid": get("uniqueID"), "mass": float(get("mass")),
                         "lat": float(get("latitude")), "lon": float(get("longitude")),
                         "area": float(get("crossSectionArea"))})
    return pd.DataFrame(rows, columns=["name", "uuid", "mass", "lat", "lon", "area"])


def _by_height(d, col):
    """(heights, values) sorted by height, for np.interp."""
    o = np.argsort(d.h.values)
    return d.h.values[o], d[col].values[o]


def darkflight_inputs(components, track, impacts, v_start=4.5):
    """Start state for dark flight of every SARA survivor, placed on the observed track.

    Pieces that lose mass start at the last point where they do. Pieces that never lose mass start
    where they have slowed to v_start (km/s), or at their last point if they never do.
    """
    rows = []
    for uuid in impacts.uuid:
        c = components.get(uuid)
        if c is None:
            continue
        d, abl = c["data"], ablation(c["data"])
        if abl.any():
            i, rule = np.flatnonzero(abl)[-1], "end of mass loss"
        else:
            slow = np.flatnonzero(d.v.values <= v_start)
            i, rule = (slow[0], f"v <= {v_start} km/s") if len(slow) else (len(d) - 1, "last SARA point")
        r = d.iloc[i]
        lat, lon, extrap = track.place(r.h, r.downrange)
        # direction of flight along the anchor's great circle at the start point
        azim = float(track.anchor.direction(lat[0], lon[0], r.downrange))
        # sphere with SARA's mass and reference area, so the area-to-mass ratio is preserved
        radius = np.sqrt(c["area"] / np.pi)
        rows.append({"component": c["name"], "uuid": uuid, "ablates": bool(abl.any()), "start_rule": rule,
                     "t_s": r.t, "h_km": r.h, "lat": lat[0], "lon": (lon[0] + 180) % 360 - 180,
                     "extrapolated": bool(extrap[0]), "v_kms": r.v, "fpa_deg": r.fpa, "azim_deg": azim,
                     "mass_kg": r.mass, "final_mass_kg": d.mass.iloc[-1], "ref_area_m2": c["area"],
                     "density_eff_kgm3": r.mass / (4 / 3 * np.pi * radius ** 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- dark flight
def calm_profile(path, top=100e3, step=250.0):
    """US Standard Atmosphere with zero wind, in OpenDarkflight's 'wrf' CSV format."""
    from opendf.Routines.Atmosphere import standardAtmosphereTemp
    h = np.arange(0, top + step, step)
    T = np.array([standardAtmosphereTemp(x) for x in h])
    g, M, R = 9.80665, 0.0289644, 8.31446
    p = 101325 * np.exp(-np.r_[0, np.cumsum(np.diff(h) * g * M / (R * 0.5 * (T[1:] + T[:-1])))])
    z = np.zeros_like(h)
    pd.DataFrame({"height": h, "temperature": T, "pressure": p, "relative_humidity": z, "wind_horizontal": z,
                  "wind_direction": z, "wind_east": z, "wind_north": z, "wind_up": z,
                  "density": p * M / (R * T)}).to_csv(path, index=False)
    return str(path)


def load_atmosphere(winds_file, winds_type, survivors, event_time, out, run_id):
    """OpenDarkflight atmosphere for the event. Returns (atm, description)."""
    from opendf.Routines.Atmosphere import AtmosphereModel, WindMeasurements

    if winds_file is None:
        path = calm_profile(out / f"{run_id}.calm_atmosphere.csv")
        print("WARNING: no wind profile given (--winds_file), dark flight uses a calm US Standard "
              "Atmosphere. Wind usually dominates the strewn-field position.")
        return AtmosphereModel(WindMeasurements(path, "wrf")), "calm US Standard Atmosphere (no wind)"

    if winds_file == "auto":
        from opendf.Darkflight import fetchAutoWinds
        from opendf.Routines.InputParams import DarkflightParameters
        # the wind wizard reads its event (place, time, height) from an OpenDarkflight input file
        ini = out / f"{run_id}.winds.ini"
        mid = survivors.iloc[len(survivors) // 2]
        ini.write_text(f"[Atmosphere]\nwinds_file = auto\nwinds_type = {winds_type}\n\n"
                       f"[Ejection]\nlat = {mid.lat:.5f}\nlon = {mid.lon:.5f}\nht = {survivors.h_km.max():.3f}\n"
                       f"ref_time = {event_time:%Y-%m-%d %H:%M:%S.%f}\n\n[Meteorite]\n")
        res = fetchAutoWinds(DarkflightParameters(str(ini)), str(ini), interactive=False)[0]
        winds_file, winds_type = res["winds_file"], res["winds_type"]

    return AtmosphereModel(WindMeasurements(winds_file, winds_type)), f"{winds_file} ({winds_type})"


def run_darkflight(survivors, atm, n_mc, sig, dh=50.0, drag_model="dfn", seed=42, wind=False, workers=None):
    """Integrate every survivor to the ground, nominal plus n_mc Monte Carlo clones, in parallel.

    Survivors with an identical start state (e.g. the reaction wheels) are integrated once and the
    result is shared: their Monte Carlo clouds come from the same distribution anyway.
    sig: 1-sigma of the start state {'pos_km', 'h_km', 'v_rel', 'angle_deg', 'mass_rel',
    'wind_speed_rel', 'wind_dir_deg'}; the wind terms only apply when a real wind profile is used.
    Returns (impacts DataFrame, {uuid: nominal Meteoroid}).
    """
    import copy
    import multiprocessing as mp
    from tqdm import tqdm
    from opendf.Routines.Engine import Meteoroid, monteCarloSample

    base = _default_params()
    base.drag_model, base.shape, base.end_ht = drag_model, "sphere", 0.0
    np.random.seed(seed)

    state = ["lat", "lon", "h_km", "v_kms", "fpa_deg", "azim_deg", "mass_kg", "density_eff_kgm3"]
    groups = survivors.groupby(survivors[state].round(6).apply(tuple, axis=1), sort=False)
    jobs, keys = [], []
    for _, members in groups:
        s = members.iloc[0]
        p = copy.copy(base)
        p.lat, p.lon, p.ht, p.vel = s.lat, s.lon, s.h_km * 1000, s.v_kms * 1000
        # OpenDarkflight takes the radiant direction: azimuth opposite to the motion, elevation > 0
        p.azim, p.alt, p.density = (s.azim_deg + 180) % 360, -s.fpa_deg, s.density_eff_kgm3
        p.lat_sigma = sig["pos_km"] / 111.2
        p.lon_sigma, p.lon_cos_corr = p.lat_sigma, True
        p.ht_sigma, p.v_sigma = sig["h_km"] * 1000, sig["v_rel"] * p.vel
        p.azim_sigma = p.alt_sigma = sig["angle_deg"]
        p.mass_sigma_rel = sig["mass_rel"]
        if wind:
            p.wind_error_mode = "statistical"
            p.wind_speed_sigma, p.wind_dir_sigma = sig["wind_speed_rel"], sig["wind_dir_deg"]
        jobs.append(Meteoroid(p, p.lat, p.lon, p.ht, p.vel, p.azim, p.alt, s.mass_kg, p.density, "sphere"))
        keys.append((members, -1))
        for k in range(n_mc):
            jobs.append(monteCarloSample(s.mass_kg, p)); keys.append((members, k))

    workers = workers or max(1, mp.cpu_count() - 1)
    print(f"\nDark flight: {len(survivors)} survivors ({groups.ngroups} distinct start states) x "
          f"(1 nominal + {n_mc} Monte Carlo) = {len(jobs)} integrations on {workers} processes")
    # the atmosphere goes to each worker once, not with every job
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(atm, dh)) as pool:
        done = list(tqdm(pool.imap(_integrate, jobs, chunksize=4), total=len(jobs), desc="dark flight"))

    rows, nominal = [], {}
    for (members, k), met in zip(keys, done):
        for _, m in members.iterrows():
            if k < 0:
                nominal[m.uuid] = met
            rows.append({"component": m.component, "uuid": m.uuid, "clone": k, "mass_kg": met.mass,
                         "lat": met.lat_data[-1], "lon": (met.lon_data[-1] + 180) % 360 - 180,
                         "fall_time_s": met.time_data[-1], "v_impact_ms": met.vel_data[-1]})
    return pd.DataFrame(rows), nominal


_WORKER = {}


def _init_worker(atm, dh):
    _WORKER.update(atm=atm, dh=dh)


def _integrate(met):
    met.integrate(_WORKER["atm"], _WORKER["dh"], False)
    return met


def _default_params():
    """OpenDarkflight DarkflightParameters holding only its defaults (read from an empty file)."""
    import tempfile
    from opendf.Routines.InputParams import DarkflightParameters
    with tempfile.TemporaryDirectory() as tmp:
        ini = Path(tmp) / "defaults.ini"
        ini.write_text("[Ejection]\n[Meteorite]\n")
        return DarkflightParameters(str(ini), file_format="opendf")


# ----------------------------------------------------------------------------- strewn field
def _local_km(lat, lon, lat0, lon0):
    """East/north km of points relative to (lat0, lon0) (equirectangular, fine over a strewn field)."""
    dlon = (np.asarray(lon) - lon0 + 180) % 360 - 180
    return dlon * np.cos(np.radians(lat0)) * 111.195, (np.asarray(lat) - lat0) * 111.195


def hull(lat, lon):
    """Convex hull of impact points -> (lat vertices, lon vertices, area km^2), closed polygon."""
    from scipy.spatial import ConvexHull, QhullError
    lat, lon = np.asarray(lat), np.asarray(lon)
    lat0, lon0 = lat.mean(), lon.mean()
    xy = np.c_[_local_km(lat, lon, lat0, lon0)]
    try:
        h = ConvexHull(xy)
    except (QhullError, ValueError):  # fewer than 3 distinct points / all on a line
        return lat, lon, 0.0
    v = np.r_[h.vertices, h.vertices[0]]
    return lat[v], (lon[v] + 180) % 360 - 180, h.volume  # 2-D 'volume' is the area


def strewnfield_summary(survivors, impacts):
    """Per-survivor impact statistics (nominal point, Monte Carlo spread along/across the track)."""
    rows = []
    for _, s in survivors.iterrows():
        imp = impacts[impacts.uuid == s.uuid]
        nom = imp[imp.clone < 0].iloc[0]
        e, n = _local_km(imp.lat, imp.lon, nom.lat, nom.lon)
        a = np.radians(s.azim_deg)
        along, cross = e * np.sin(a) + n * np.cos(a), e * np.cos(a) - n * np.sin(a)
        _, _, area = hull(imp.lat, imp.lon)
        rows.append({"component": s.component, "uuid": s.uuid, "mass_kg": s.mass_kg,
                     "start_h_km": s.h_km, "start_lat": s.lat, "start_lon": s.lon, "start_rule": s.start_rule,
                     "impact_lat": nom.lat, "impact_lon": nom.lon, "fall_time_s": nom.fall_time_s,
                     "v_impact_ms": nom.v_impact_ms,
                     "sigma_along_km": along.std(), "sigma_cross_km": cross.std(), "hull_area_km2": area})
    return pd.DataFrame(rows)


def write_strewnfield(summary, impacts, run_id, settings, out):
    """Readable text report + KML of the strewn field; returns the overall hull."""
    import simplekml
    hlat, hlon, area = hull(impacts.lat, impacts.lon)
    lines = [f"Strewn field of the SARA survivors - {run_id}", "=" * 72, ""]
    lines += [f"{k:<22}: {v}" for k, v in settings.items()]
    lines += ["", f"Impact points        : {len(impacts)} ({len(summary)} survivors, nominal + Monte Carlo)",
              f"Strewn-field area    : {area:.1f} km^2 (convex hull of all impact points)",
              f"Latitude range       : {impacts.lat.min():.4f} .. {impacts.lat.max():.4f} deg",
              f"Longitude range      : {impacts.lon.min():.4f} .. {impacts.lon.max():.4f} deg",
              f"Centre (mean)        : {impacts.lat.mean():.4f}, {impacts.lon.mean():.4f} deg",
              "", "Per survivor (nominal impact, 1-sigma Monte Carlo spread along/across the track):", ""]
    cols = ["component", "mass_kg", "start_h_km", "start_rule", "impact_lat", "impact_lon",
            "fall_time_s", "sigma_along_km", "sigma_cross_km", "hull_area_km2"]
    lines += summary[cols].round(4).to_string(index=False).splitlines()
    lines += ["", "Strewn-field polygon (convex hull, closed; lat, lon in deg):", ""]
    lines += [f"{la:11.5f}, {lo:11.5f}" for la, lo in zip(hlat, hlon)]
    (out / f"{run_id}.strewnfield.txt").write_text("\n".join(lines) + "\n")

    kml = simplekml.Kml(name=f"{run_id} strewn field")
    pol = kml.newpolygon(name=f"strewn field ({area:.0f} km2)", outerboundaryis=list(zip(hlon, hlat)))
    pol.style.polystyle.color = simplekml.Color.changealphaint(70, simplekml.Color.red)
    for _, s in summary.iterrows():
        kml.newpoint(name=f"{s.component} ({s.mass_kg:.3g} kg)", coords=[(s.impact_lon, s.impact_lat)])
    kml.save(str(out / f"{run_id}.strewnfield.kml"))
    return hlat, hlon, area


def plot_strewnfield(survivors, impacts, nominal, segments, stations, track, strewn, out):
    """Observed track + dark-flight start points (left) and a zoom on the strewn field (right)."""
    obs = pd.concat(segments.values())
    lon_ref = impacts.lon.mean()
    unwrap = lambda lon: (np.asarray(lon) - lon_ref + 180) % 360 - 180 + lon_ref
    names = sorted(survivors.component.unique())
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    col = {n: cmap(i % cmap.N) for i, n in enumerate(names)}

    fig = plt.figure(figsize=(17, 8))
    axes = []
    for k in range(2):
        if ccrs:
            ax = fig.add_subplot(1, 2, k + 1, projection=ccrs.PlateCarree(central_longitude=lon_ref))
            ax.add_feature(cfeature.LAND, color="0.93")
            ax.add_feature(cfeature.COASTLINE, lw=0.6)
            ax.add_feature(cfeature.BORDERS, lw=0.3)
            ax.gridlines(draw_labels=True, lw=0.3)
        else:
            ax = fig.add_subplot(1, 2, k + 1)
            ax.grid(alpha=0.3)
            ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
        axes.append(ax)
    tr = {"transform": ccrs.PlateCarree()} if ccrs else {}

    def extent(ax, lat, lon, pad):
        lon = unwrap(lon)
        e = [lon.min() - pad, lon.max() + pad, np.min(lat) - pad, np.max(lat) + pad]
        if ccrs:
            ax.set_extent(e, crs=ccrs.PlateCarree())
        else:
            ax.set_xlim(e[:2]); ax.set_ylim(e[2:])
        scale_bar(ax)

    # --- overview: observations, extension of the observed track, start points, strewn field
    ax = axes[0]
    sc = ax.scatter(unwrap(obs.lon), obs.lat, c=obs.absmag, cmap="inferno_r", s=6, zorder=5, **tr)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Observed absolute magnitude")
    for s, (la, lo) in stations.items():
        ax.plot(unwrap(lo), la, "^", color="tab:blue", ms=5, **tr)
    an = track.anchor
    far = gc_distance(an.lat, an.lon, impacts.lat.values, impacts.lon.values).max() + 50
    glat, glon = destination(an.lat, an.lon, an.azim, np.linspace(0, far, 200))
    ax.plot(unwrap(glon), glat, "--", color="0.4", lw=1, label="observed track, extended from the anchor", **tr)
    ax.plot(unwrap(an.lon), an.lat, "*", color="tab:blue", mec="k", ms=13, zorder=7,
            label=f"anchor ({an.segment}, {an.h:.2f} km)", **tr)
    ax.plot(unwrap(survivors.lon), survivors.lat, "*", color="red", ms=9, zorder=6,
            label="dark-flight start points", **tr)
    ax.fill(unwrap(strewn[1]), strewn[0], color="red", alpha=0.3, label="strewn field", **tr)
    ax.plot([], [], "^", color="tab:blue", label="stations")
    extent(ax, np.r_[obs.lat, impacts.lat], np.r_[obs.lon, impacts.lon], 2)
    ax.set_title("Observed re-entry and dark-flight start points")
    ax.legend(fontsize=8, loc="upper left")

    # --- zoom on the strewn field
    ax = axes[1]
    ax.fill(unwrap(strewn[1]), strewn[0], color="red", alpha=0.12, **tr)
    ax.plot(unwrap(strewn[1]), strewn[0], color="red", lw=1, label=f"strewn field ({strewn[2]:.0f} km$^2$)", **tr)
    for n in names:
        imp = impacts[impacts.component == n]
        ax.plot(unwrap(imp.lon[imp.clone >= 0]), imp.lat[imp.clone >= 0], ".", ms=2, alpha=0.5, color=col[n], **tr)
    for _, s in survivors.drop_duplicates("component").iterrows():
        met = nominal[s.uuid]
        ax.plot(unwrap(met.lon_data), met.lat_data, "-", lw=0.8, color=col[s.component], **tr)
        imp = impacts[(impacts.uuid == s.uuid) & (impacts.clone < 0)].iloc[0]
        ax.plot(unwrap(imp.lon), imp.lat, "o", mec="k", mew=0.6, ms=7, color=col[s.component],
                label=f"{s.component.replace('_', ' ')} ({s.mass_kg:.3g} kg)", **tr)
    ax.plot(unwrap(survivors.lon), survivors.lat, "*", color="red", ms=9, zorder=6, **tr)
    zlat = np.r_[impacts.lat, survivors.lat]
    zlon = np.r_[impacts.lon, survivors.lon]
    extent(ax, zlat, zlon, 0.1 * max(np.ptp(zlat), np.ptp(unwrap(zlon)), 0.5))
    ax.set_title("Strewn field (lines: nominal dark flight from the start star; dots: Monte Carlo)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_lightcurve_check(components, segments, recorded, chain, anchor, out):
    """Altitude vs. downrange of every SARA component with the observations laid on top.

    All observations are tied to SARA by the one anchor: each point takes the anchor's SARA
    downrange plus its along-track distance from the anchor. Brightness (AbsMag) is compared below
    with SARA's mass-loss rate of the components losing mass in that segment.
    """
    segs = list(segments)
    rec_groups = set(recorded.group) if len(recorded) else set()
    colors = group_colors(rec_groups)
    fig = plt.figure(figsize=(6 * len(segs), 13))
    gs = fig.add_gridspec(3, len(segs), height_ratios=[1.1, 1, 0.8])
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(chain.downrange, chain.h, color="k", lw=1.5, label="parent body (compound)")

    def draw_sara(ax, lw_scale=1.0):
        for c in components.values():
            d, abl = c["data"], ablation(c["data"])
            ax.plot(d.downrange, d.h, color="0.75", lw=0.6 * lw_scale, zorder=1)
            if abl.any():
                g = group_name(c["name"])
                ax.plot(d.downrange.where(abl), d.h.where(abl), color=colors.get(g, "0.45"),
                        lw=(2.2 if g in rec_groups else 1.0) * lw_scale, zorder=2)

    draw_sara(ax0)
    for g in sorted(rec_groups):
        ax0.plot([], [], color=colors[g], lw=2.2, label=g)
    ax0.plot([], [], color="0.45", label="other components losing mass")
    ax0.plot([], [], color="0.75", label="SARA components (no mass loss)")

    norm = plt.Normalize(pd.concat(segments.values()).absmag.min(), pd.concat(segments.values()).absmag.max())
    placed = {}
    for k, seg in enumerate(segs):
        obs = segments[seg].sort_values("length")
        rec = recorded[recorded.segment == seg] if len(recorded) else recorded
        uuids = list(rec.uuid.unique()) if len(rec) else []
        dr = anchor.downrange_of(obs.lat.values, obs.lon.values)
        placed[seg] = dr
        sc = ax0.scatter(dr, obs.h, c=obs.absmag, cmap="inferno_r", norm=norm, s=6, zorder=4)
        ax0.annotate(seg, (dr.min(), obs.h.max()), xytext=(0, 8), textcoords="offset points", fontsize=9)

        # zoom on this segment
        ax = fig.add_subplot(gs[1, k])
        draw_sara(ax, 1.5)
        ax.scatter(dr, obs.h, c=obs.absmag, cmap="inferno_r", norm=norm, s=10, zorder=4)
        pad_x, pad_h = max(2.0, 0.15 * np.ptp(dr)), max(0.5, 0.3 * np.ptp(obs.h))
        ax.set_xlim(dr.min() - pad_x, dr.max() + pad_x)
        ax.set_ylim(obs.h.min() - pad_h, obs.h.max() + pad_h)
        if seg == anchor.segment:
            ax.plot(anchor.downrange, anchor.h, "*", color="tab:blue", ms=14, mec="k", zorder=5,
                    label=f"anchor ({anchor.h:.2f} km)")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{seg}: {len(uuids)} SARA components losing mass here", fontsize=10)
        ax.set_xlabel("Downrange [km]")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("Altitude [km]")

        # light curve vs SARA mass-loss rate, on the same downrange axis
        axm = fig.add_subplot(gs[2, k], sharex=ax)
        axm.scatter(dr, obs.absmag, c=obs.absmag, cmap="inferno_r", norm=norm, s=6)
        axm.invert_yaxis()
        axm.set_xlabel("Downrange [km]")
        axm.grid(alpha=0.3)
        if k == 0:
            axm.set_ylabel("Absolute magnitude")
        if uuids:
            axr = axm.twinx()
            grid = np.linspace(*ax.get_xlim(), 400)
            rate = np.zeros_like(grid)
            for u in uuids:
                d = components[u]["data"]
                mdot = np.clip(-np.gradient(d.mass.values, d.t.values), 0, None)
                inside = (grid >= d.downrange.min()) & (grid <= d.downrange.max())
                rate[inside] += np.interp(grid[inside], d.downrange.values, mdot)
            axr.plot(grid, rate, color="tab:green", lw=1.5)
            axr.set_ylabel("SARA mass-loss rate [kg/s]", color="tab:green")

    ax0.plot(anchor.downrange, anchor.h, "*", color="tab:blue", ms=14, mec="k", zorder=5,
             label=f"anchor ({anchor.segment}, {anchor.h:.2f} km)")
    fig.colorbar(sc, ax=ax0, pad=0.01, label="Observed absolute magnitude")
    ax0.set_xlabel("SARA downrange [km]")
    ax0.set_ylabel("Altitude [km]")
    ax0.set_title("SARA altitude vs downrange\nobservations placed by their along-track distance "
                  "from the anchor", fontsize=10)
    lo = min(o.h.min() for o in segments.values())
    ax0.set_ylim(max(0, lo - 25), max(o.h.max() for o in segments.values()) + 10)
    x = np.concatenate(list(placed.values()))
    ax0.set_xlim(x.min() - 400, x.max() + 400)
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_velocity_check(components, segments, recorded, chain, out):
    """Observed speed vs. SARA speed against altitude, one panel per observed segment.

    The observed 'Vel' of the report is a point-to-point speed and very noisy, so it is drawn as a
    scatter first; the SARA speed of the components losing mass inside that segment (from their
    Trajectory files) is drawn on top. The median observed - SARA difference is given per segment.
    """
    segs = list(segments)
    fig, axes = plt.subplots(1, len(segs), figsize=(6 * len(segs), 7), squeeze=False)
    rows = []
    for ax, seg in zip(axes[0], segs):
        obs = segments[seg][segments[seg].v > 0]  # first point of each station has no speed
        rec = recorded[recorded.segment == seg] if len(recorded) else recorded
        uuids = list(rec.uuid.unique()) if len(rec) else []
        for k, (st, o) in enumerate(obs.groupby("station")):
            ax.scatter(o.v, o.h, s=8, alpha=0.5, color=plt.get_cmap("tab10")(k), zorder=1,
                       label=f"observed {st}")

        refs = [(components[u]["data"], group_name(components[u]["name"])) for u in uuids]
        colors = group_colors({g for _, g in refs})
        seen = set()
        for d, g in refs:
            ax.plot(d.v, d.h, color=colors[g], lw=1.8, zorder=3, label=g if g not in seen else None)
            seen.add(g)
        ax.plot(chain.v, chain.h, "--", color="k", lw=1.2, zorder=2, label="parent body (compound)")

        # SARA speed at the observed heights: median over the matched components (parent if none)
        sim = np.median([np.interp(obs.h, *_by_height(d, "v")) for d, _ in refs]
                        or [np.interp(obs.h, *_by_height(chain, "v"))], axis=0)
        dv = obs.v.values - sim
        med, scat = np.median(dv), 1.4826 * np.median(np.abs(dv - np.median(dv)))
        rows.append((seg, med, scat, len(obs)))
        ax.text(0.02, 0.02, f"median obs - SARA: {med * 1000:+.0f} m/s\n"
                            f"robust scatter: {scat * 1000:.0f} m/s ({len(obs)} points)",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})

        pad_h = max(0.5, 0.1 * np.ptp(obs.h))
        ax.set_ylim(obs.h.min() - pad_h, obs.h.max() + pad_h)
        # noisy outliers would squash the curves: limit to the bulk of the observations + SARA
        lo, hi = np.percentile(obs.v, [2, 98])
        lo, hi = min(lo, sim.min()), max(hi, sim.max())
        ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_title(f"{seg}: {len(uuids)} SARA components losing mass here", fontsize=10)
        ax.set_xlabel("Speed [km/s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0][0].set_ylabel("Altitude [km]")
    fig.suptitle("Observed point-to-point speed vs. SARA speed at the same altitudes")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sara_dir", default=r"C:\Users\maxiv\Documents\UWO\Re-entry\20250622\DRAMA-SARA_V1")
    ap.add_argument("--reports", default=[r"C:\Users\maxiv\Documents\UWO\Re-entry\20250622\skyfit_traj"], nargs="+", help="*_report.txt files or folders containing them")
    ap.add_argument("--out", default=r"C:\Users\maxiv\Documents\UWO\Re-entry\20250622\Wind", help="output folder (default: SARA folder)")
    ap.add_argument("--curved", action="store_true", help="use *_curved_report.txt")
    ap.add_argument("--anchor_segment", default=None,
                    help="observed segment to anchor SARA to (default: lowest well-constrained one)")
    ap.add_argument("--anchor_min_qconv", type=float, default=10.0,
                    help="minimum best convergence angle [deg] of an anchor segment (default 10)")
    g = ap.add_argument_group("dark flight / strewn field (OpenDarkflight)")
    g.add_argument("--v_start", type=float, default=4.5,
                   help="pieces that never lose mass start dark flight at this speed [km/s] (default 4.5)")
    g.add_argument("--winds_file", default='auto',
                   help="wind profile for the dark flight; 'auto' downloads one with the OpenDarkflight "
                        "wizard; omitted = calm US Standard Atmosphere")
    g.add_argument("--winds_type", default=None,
                   help="format of --winds_file (wrf, wyoming; default wrf) or, with --winds_file auto, "
                        "the wizard directive (default auto:best; e.g. auto:gfs, auto:era5, auto:radiosonde)")
    g.add_argument("--mc", type=int, default=100, help="Monte Carlo clones per survivor (default 100, 0 = off)")
    g.add_argument("--drag_model", default="dfn", choices=["dfn", "ceplecha", "loth", "constant"])
    g.add_argument("--dh", type=float, default=50.0,
                   help="integration height step [m] (default 50; 100+ goes unstable for light, slow pieces)")
    g.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPU cores - 1)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--sig_pos_km", type=float, default=5.0, help="1-sigma start position [km]")
    g.add_argument("--sig_h_km", type=float, default=1.0, help="1-sigma start height [km]")
    g.add_argument("--sig_v_rel", type=float, default=0.05, help="1-sigma start speed, fraction")
    g.add_argument("--sig_angle_deg", type=float, default=1.0, help="1-sigma azimuth and flight-path angle [deg]")
    g.add_argument("--sig_mass_rel", type=float, default=0.2, help="1-sigma mass, fraction")
    g.add_argument("--sig_wind_speed_rel", type=float, default=0.2, help="1-sigma wind speed, fraction")
    g.add_argument("--sig_wind_dir_deg", type=float, default=10.0, help="1-sigma wind direction [deg]")
    args = ap.parse_args()
    out = Path(args.out or args.sara_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_id, components, chain = load_sara(args.sara_dir)
    reports = find_reports(args.reports, args.curved)
    if not reports:
        raise SystemExit("no *_report.txt found")
    segments, stations = {}, {}
    for r in reports:
        seg, st = load_report(r)
        segments[r.parent.parent.name if r.parent.parent.name.startswith("skyfit") else r.stem] = seg
        stations.update(st)
    obs = pd.concat(segments.values())

    recorded = match(components, segments)
    recorded.to_csv(out / f"{run_id}.recorded_components.csv", index=False)

    # one anchor ties SARA to the sky; everything below it is placed from there
    anchor, candidates = choose_anchor(components, recorded, segments, chain,
                                       min_qconv=args.anchor_min_qconv, segment=args.anchor_segment)
    candidates.to_csv(out / f"{run_id}.anchor_candidates.csv", index=False)
    track = ObservedTrack(obs, chain, anchor)
    check = anchor_check(anchor, segments)
    print(f"Anchor: {anchor.describe()}")

    impacts = load_impacts(args.sara_dir)
    survivors = darkflight_inputs(components, track, impacts, v_start=args.v_start)
    survivors.to_csv(out / f"{run_id}.darkflight_inputs.csv", index=False)

    plot_altitude_downrange(components, segments, recorded, chain, track,
                            out / f"{run_id}.recorded_AltitudeVsDownrange.png")
    plot_ground_map(components, segments, recorded, stations, chain, track, impacts, survivors,
                    out / f"{run_id}.recorded_GroundMap.png")
    plot_lightcurve_check(components, segments, recorded, chain, anchor, out / f"{run_id}.lightcurve_check.png")
    speeds = plot_velocity_check(components, segments, recorded, chain, out / f"{run_id}.velocity_check.png")

    # --- dark flight of the survivors -> strewn field
    event_time = datetime(2000, 1, 1, 12) + timedelta(days=obs.jd.max() - 2451545.0)
    winds_type = args.winds_type or ("auto:best" if args.winds_file == "auto" else "wrf")
    atm, wind_desc = load_atmosphere(args.winds_file, winds_type, survivors, event_time, out, run_id)
    sig = {"pos_km": args.sig_pos_km, "h_km": args.sig_h_km, "v_rel": args.sig_v_rel,
           "angle_deg": args.sig_angle_deg, "mass_rel": args.sig_mass_rel,
           "wind_speed_rel": args.sig_wind_speed_rel, "wind_dir_deg": args.sig_wind_dir_deg}
    df_impacts, nominal = run_darkflight(survivors, atm, args.mc, sig, dh=args.dh, drag_model=args.drag_model,
                                         seed=args.seed, wind=args.winds_file is not None,
                                         workers=args.workers)
    df_impacts.to_csv(out / f"{run_id}.strewnfield_impacts.csv", index=False)
    summary = strewnfield_summary(survivors, df_impacts)
    settings = {"SARA run": args.sara_dir, "event time (UTC)": f"{event_time:%Y-%m-%d %H:%M:%S}",
                "anchor": anchor.describe(),
                "dark-flight start": f"end of mass loss; non-ablating pieces at v <= {args.v_start} km/s",
                "atmosphere / wind": wind_desc, "drag model": f"{args.drag_model}, sphere with SARA mass/area",
                "Monte Carlo": f"{args.mc} clones per survivor, seed {args.seed}",
                "1-sigma": ", ".join(f"{k}={v}" for k, v in sig.items()
                                     if args.winds_file is not None or not k.startswith("wind"))}
    strewn = write_strewnfield(summary, df_impacts, run_id, settings, out)
    plot_strewnfield(survivors, df_impacts, nominal, segments, stations, track, strewn,
                     out / f"{run_id}.strewnfield_map.png")

    # --- summary
    print(f"{run_id}: {len(components)} components, "
          f"{sum(ablation(c['data']).any() for c in components.values())} lose mass")
    for seg, o in segments.items():
        print(f"  {seg}: {len(o)} points, {o.h.max():.1f} -> {o.h.min():.1f} km, AbsMag {o.absmag.min():+.2f}..{o.absmag.max():+.2f}")
    obs_len = gc_distance(*track.first, anchor.lat, anchor.lon)
    print(f"  ground distance {track.h.max():.1f}->{anchor.h:.2f} km (top -> anchor): observed {obs_len:.0f} km, "
          f"SARA parent body {anchor.downrange - track.d_top:.0f} km")
    print("\nAnchor candidates:")
    print(candidates.round(3).to_string(index=False) if len(candidates) else "  none (parent body)")
    print(f"Anchor: {anchor.describe()}")
    print("\nIndependent check, observations past the anchor vs. the anchored SARA reference at the same "
          "position (dh > 0: observed higher than SARA):")
    print(check.round(2).to_string(index=False) if len(check) else "  no observations past the anchor")
    if len(recorded):
        print("\nRecorded (losing mass inside an observed altitude range):")
        summ = recorded.groupby(["segment", "group"]).agg(n=("uuid", "nunique"), mass_lost_kg=("mass_lost_kg", "sum"),
                                                           h_start=("h_start_km", "max"), h_end=("h_end_km", "min"))
        print(summ.round(2).to_string())
    print("\nSpeed check (observed - SARA at the observed heights):")
    for seg, dv, scat, n in speeds:
        print(f"  {seg}: median {dv * 1000:+.0f} m/s, robust scatter {scat * 1000:.0f} m/s ({n} points)")
    print(f"\nSurvivors / dark-flight start states ({len(survivors)}):")
    print(survivors[["component", "start_rule", "h_km", "lat", "lon", "v_kms", "mass_kg"]].round(3).to_string(index=False))
    print(f"\nStrewn field ({wind_desc}): {strewn[2]:.1f} km^2")
    print(summary[["component", "mass_kg", "impact_lat", "impact_lon", "fall_time_s",
                   "sigma_along_km", "sigma_cross_km"]].round(3).to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
