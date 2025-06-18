import os
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, FK5, SkyCoord
from astropy import units as u
import sys
import math

@dataclass
class MirSite:
    event: str = ''
    site: int = 0
    ts: int = 0
    tu: int = 0
    lat: float = 0.0
    lon: float = 0.0
    elev: float = 0.0
    th0: float = 0.0
    phi0: float = 0.0
    rot: float = 0.0
    wid: int = 0
    ht: int = 0
    offset: float = 0
    site_name: str = ''
    path: str =''
    starcat: int = 0

@dataclass
class MeteorMir:
    site: int = 0
    type: str = ''
    fr: int = 0
    id: int = 0
    cx: float = 0.0
    cy: float = 0.0
    th: float = 0.0
    phi: float = 0.0
    tau: float = 0.0
    lsp: float = 0.0
    cr: int = 0
    cg: int = 0
    cb: int = 0
    ts: int = 0
    tu: int = 0
    # Additional fields if needed
    t: float = 0.0
    L: float = 0.0
    R: float = 0.0
    ht: float = 0.0
    vel: float = 0.0
    rv: float = 0.0
    rh: float = 0.0
    lat: float = 0.0
    lon: float = 0.0

def find_keyword_index(tokens: list, keyword: str) -> int:
    """Return the index of 'keyword' in 'tokens' or -1 if not found."""
    try:
        return tokens.index(keyword)
    except ValueError:
        return -1

def maybe_convert_radians_to_degrees(value: float) -> float:
    """If |value|<3.2, treat as radians => convert to degrees. Else keep as is."""
    if abs(value) < 3.2:
        return math.degrees(value)
    return value

def parse_site_line(tokens: list, mir_site: MirSite, label: str):
    """
    Parse site data into 'mir_site'. 'label' is just to help debug which site
    object we're updating (e.g. 'site1', 'site2').
    """
    idx_site = find_keyword_index(tokens, 'site')
    if idx_site != -1 and (idx_site+1) < len(tokens):
        mir_site.site = int(tokens[idx_site + 1])

    idx_ts = find_keyword_index(tokens, 'ts')
    if idx_ts != -1 and (idx_ts+1) < len(tokens):
        mir_site.ts = int(float(tokens[idx_ts + 1]))

    idx_tu = find_keyword_index(tokens, 'tu')
    if idx_tu != -1 and (idx_tu+1) < len(tokens):
        mir_site.tu = int(float(tokens[idx_tu + 1]))

    # lat
    idx_lat = find_keyword_index(tokens, 'lat')
    if idx_lat != -1 and (idx_lat + 1) < len(tokens):
        val = float(tokens[idx_lat + 1])
        mir_site.lat = maybe_convert_radians_to_degrees(val)

    # lon
    idx_lon = find_keyword_index(tokens, 'lon')
    if idx_lon != -1 and (idx_lon + 1) < len(tokens):
        val = float(tokens[idx_lon + 1])
        mir_site.lon = maybe_convert_radians_to_degrees(val)

    # elev
    idx_elv = find_keyword_index(tokens, 'elv')
    if idx_elv != -1 and (idx_elv + 1) < len(tokens):
        mir_site.elev = float(tokens[idx_elv + 1])

    # th0
    idx_th0 = find_keyword_index(tokens, 'th0')
    if idx_th0 != -1 and (idx_th0 + 1) < len(tokens):
        val = float(tokens[idx_th0 + 1])
        mir_site.th0 = maybe_convert_radians_to_degrees(val)

    # phi0
    idx_phi0 = find_keyword_index(tokens, 'phi0')
    if idx_phi0 != -1 and (idx_phi0 + 1) < len(tokens):
        val = float(tokens[idx_phi0 + 1])
        mir_site.phi0 = maybe_convert_radians_to_degrees(val)

    idx_offset = find_keyword_index(tokens, 'offset')
    if idx_offset != -1 and (idx_offset + 1) < len(tokens):
        mir_site.offset = float(tokens[idx_offset + 1])

    idx_rotate = find_keyword_index(tokens, 'rotate')
    if idx_rotate != -1 and (idx_rotate + 1) < len(tokens):
        mir_site.rot = float(tokens[idx_rotate + 1])

    # optional
    idx_sitename = find_keyword_index(tokens, 'sitename')
    if idx_sitename != -1 and (idx_sitename + 1) < len(tokens):
        mir_site.site_name = tokens[idx_sitename + 1].strip("'")

    idx_starcat = find_keyword_index(tokens, 'starcat')
    if idx_starcat != -1 and (idx_starcat + 1) < len(tokens):
        mir_site.starcat = tokens[idx_starcat + 1].strip("'")

    idx_wid = find_keyword_index(tokens, 'wid')
    if idx_wid != -1 and (idx_wid + 1) < len(tokens) and mir_site.wid==0:
        mir_site.wid = int(float(tokens[idx_wid + 1]))

    idx_ht = find_keyword_index(tokens, 'ht')
    if idx_ht != -1 and (idx_ht + 1) < len(tokens) and mir_site.ht==0:
        mir_site.ht = int(float(tokens[idx_ht + 1]))
    
    ind_path = find_keyword_index(tokens, 'path')
    if ind_path != -1 and (ind_path + 1) < len(tokens):
        mir_site.path = tokens[ind_path + 1].strip("'")
        # take the file name
        mir_site.event = mir_site.path.split(os.sep)[-1]
        # take anyhtn before the dot
        mir_site.event = mir_site.path.split('.')[0]

    # Debug print
    # print(f"DEBUG parse_site_line => {label}: site={mir_site.site}, phi0={mir_site.phi0}, th0={mir_site.th0}")

def read_mirfit_state_file_data(filepath: str) -> Tuple[
    list, str, np.ndarray, np.ndarray, np.ndarray, float, MirSite, MirSite
]:
    mir_site1 = MirSite()
    mir_site2 = MirSite()
    meteor_list = []
    st_arr, seq_arr, ts_arr, tu_arr = [], [], [], []

    dir_path = os.path.dirname(filepath)
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return meteor_list, "", np.array([]), np.array([]), np.array([0.0]), 0.0, mir_site1, mir_site2

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line_clean = line.replace(';',' ')
        tokens = line_clean.split()
        if not tokens:
            continue

        # # Debug
        # if tokens[0].lower() in ['video','plate','exact']:
        #     print(f"DEBUG => {tokens[0].lower()} line tokens:", tokens)
        #     print("DEBUG => site1 before parse:", mir_site1.phi0, mir_site1.th0)
        #     print("DEBUG => site2 before parse:", mir_site2.phi0, mir_site2.th0)

        # video/plate/exact => parse
        if tokens[0].lower() in ['video','plate','exact','scale']:
            idx_site = find_keyword_index(tokens, 'site')
            if idx_site != -1 and (idx_site+1)<len(tokens):
                site_str = tokens[idx_site+1]
                # print('DEBUG => ',mir_site1)
                # print('DEBUG => ',mir_site2)
                # print("DEBUG => Found site_str:", site_str, tokens)
                if site_str in ['1','4']:
                    parse_site_line(tokens, mir_site1, label="site1")
                elif site_str == '2':
                    parse_site_line(tokens, mir_site2, label="site2")
                else:
                    print(f"WARNING: site_str is '{site_str}' not recognized as '1','2','4'")

            continue

        # frame
        if tokens[0].lower() == 'frame' and len(tokens)>13:
            try:
                st_arr.append(int(tokens[3]))
                ts_arr.append(int(tokens[9]))
                tu_arr.append(int(tokens[11]))
                seq_arr.append(int(tokens[13]))
            except ValueError:
                print(f"Error parsing frame line: {line}")
            continue

        # mark => meteor
        if tokens[0].lower() == 'mark':
            if "type" in tokens:
                idx_type = tokens.index("type")
                if idx_type+1<len(tokens) and tokens[idx_type+1].lower()=="meteor":
                    m = MeteorMir()
                    i = 1
                    while i<(len(tokens)-1):
                        k = tokens[i]
                        v = tokens[i+1]
                        if k=="site":
                            m.site = int(v)
                        elif k=="cx":
                            m.cx = float(v)
                        elif k=="cy":
                            m.cy = float(v)
                        elif k=="th":
                            m.th = float(v)
                        elif k=="phi":
                            m.phi= float(v)
                        elif k=="ts":
                            m.ts = int(float(v))
                        elif k=="tu":
                            m.tu = int(float(v))
                        elif k=="lsp":
                            try: m.lsp = float(v)
                            except: m.lsp = np.nan
                        i+=2
                    meteor_list.append(m)
            continue

    # end for lines

    st_arr_np = np.array(st_arr, dtype=int)
    seq_arr_np= np.array(seq_arr, dtype=int)
    ts_arr_np = np.array(ts_arr, dtype=int)
    tu_arr_np = np.array(tu_arr, dtype=int)

    valid = ts_arr_np > 0
    st_arr_np = st_arr_np[valid]
    seq_arr_np= seq_arr_np[valid]
    ts_arr_np = ts_arr_np[valid]
    tu_arr_np = tu_arr_np[valid]

    if len(ts_arr_np)>0 and len(meteor_list)>0:
        time0 = ts_arr_np + tu_arr_np/1e6
        rel_time = time0 - time0[0]
        ts0 = ts_arr_np[0]
    else:
        rel_time = np.array([0.0])
        ts0 = 0.0

    return meteor_list, dir_path, st_arr_np, seq_arr_np, rel_time, ts0, mir_site1, mir_site2

def hor2eq(alt: float, az: float, obstime: Time, location: EarthLocation) -> Tuple[float,float]:
    altaz = AltAz(alt=alt*u.deg, az=az*u.deg, obstime=obstime, location=location)
    eq = altaz.transform_to(FK5())
    return eq.ra.deg, eq.dec.deg

def hor2eq_astropy(alt_deg: float, az_deg: float, obs_time: Time, location: EarthLocation, pressure_hpa: float = 1013.25, temperature_c: float = 15.0) -> Tuple[float, float]:
    """
    Convert (alt_deg, az_deg) -> (ra_deg, dec_deg) in J2000, applying
    atmospheric refraction, nutation, aberration, and precession via Astropy.

    pressure_hpa, temperature_c => typical atmospheric conditions for refraction.
    """
    # Build an AltAz frame that includes refraction
    altaz_frame = AltAz(
        obstime=obs_time,
        location=location,
        pressure=pressure_hpa * u.hPa,     # e.g. 1013.25
        temperature=temperature_c * u.deg_C, # e.g. 15°C
        relative_humidity=0,
        obswl=0.5 * u.um  # typical visible wavelength
    )
    # Create a SkyCoord in the altaz_frame
    altaz_coord = SkyCoord(
        alt=alt_deg * u.deg,
        az=az_deg * u.deg,
        frame=altaz_frame
    )
    # Transform to J2000 coordinates
    eq_j2000 = altaz_coord.transform_to(FK5(equinox='J2000'))
    return eq_j2000.ra.deg, eq_j2000.dec.deg


def generate_ecsv_header(mir_site: MirSite, camera_id: str, event: str, out_dir: str) -> str:

    # cat_char = event[22] if len(event) > 22 else ' '
    cat_char = event[-1]
    if cat_char in ['A','T','K']:
        cat = 'STARS9TH_VBVRI R band'
    elif cat_char in ['G','F']:
        cat = 'GAIA G band'
    else:
        cat = 'STARS9TH_VBVRI R band'

    # origin = event[24:27] if len(event)>23 else 'MET'
    # if len(camera_id)>=2 and camera_id[1]=='T':
    #     origin='Mirfit'
    origin = 'Mirfit' if camera_id=='T' else 'MET'
    camera_id='0'+str(mir_site.site)+camera_id

    obs_az = 360.0-(mir_site.phi0-90.0)  # 450 - phi0
    obs_ev = 90.0 - mir_site.th0
    obs_rot= mir_site.rot
    if len(event)>22 and event[22]=='A':
        obs_rot=0.0

    file_name = out_dir+os.sep+event.split('_')[1][0:4]+'-'+event.split('_')[1][4:6]+'-'+event.split('_')[1][6:8]+'T'+event.split('_')[2][0:2]+'_'+event.split('_')[2][2:4]+'_'+event.split('_')[2][4:6]+'_'+origin+'_'+camera_id+'.ecsv'
    # astropy-{astropy.__version__}

    # iso_start = event[:19] if len(event)>=19 else ''
    # take all the numbers of and separate ev_20240713_031407A_02T to get 2024-07-13T03:14:07.879222
    iso_start = event.split('_')[1][0:4]+'-'+event.split('_')[1][4:6]+'-'+event.split('_')[1][6:8]+'T'+event.split('_')[2][0:2]+':'+event.split('_')[2][2:4]+':'+event.split('_')[2][4:6]+'.'+str(mir_site.tu)
    # now put year as the first 4 of iso_start then '-' then month then
    header = f"""# %ECSV 0.9
# ---
# datatype:
# - {{name: datetime, datatype: string}}
# - {{name: ra, unit: deg, datatype: float64}}
# - {{name: dec, unit: deg, datatype: float64}}
# - {{name: azimuth, datatype: float64}}
# - {{name: altitude, datatype: float64}}
# - {{name: x_image, unit: pix, datatype: float64}}
# - {{name: y_image, unit: pix, datatype: float64}}
# - {{name: integrated_pixel_value, datatype: int64}}
# - {{name: mag_data, datatype: float64}}
# delimiter: ','
# meta: !!omap
# - {{obs_latitude: {mir_site.lat:.4f}}}
# - {{obs_longitude: {mir_site.lon:.4f}}}
# - {{obs_elevation: {mir_site.elev:.4f}}}
# - {{origin: '{origin}'}}
# - {{camera_id: '{camera_id}'}}
# - {{cx: {mir_site.wid}}}
# - {{cy: {mir_site.ht}}}
# - {{photometric_band: '{cat}'}}
# - {{image_file: '{event}.vid'}}
# - {{isodate_start_obs: '{iso_start}'}}
# - {{astrometry_number_stars: 52}}
# - {{mag_label: 'mag_data'}}
# - {{no_frags: 1}}
# - {{obs_az: {obs_az:.14f}}}
# - {{obs_ev: {obs_ev:.14f}}}
# - {{obs_rot: {obs_rot:.14f}}}
# - {{fov_horiz: 14.754961247941647}}
# - {{fov_vert: 14.752316747496725}}
# schema: astropy-2.0
datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value,mag_data
"""
    return header,file_name

def process_site_old(meteors: list, mir_site: MirSite, location: EarthLocation, out_dir: str):
    if not meteors:
        print(f"No meteor data found.")
        return

    # camera_id = mir_site.event[20:23] if len(mir_site.event)>23 else "Cam"
    camera_id = mir_site.event[-1]
    hdr, out_file = generate_ecsv_header(mir_site, camera_id, mir_site.event, out_dir)
    with open(out_file, "w") as f:
        f.write(hdr)

    lines=[]
    for m in meteors:
        ts_total = m.ts + m.tu/1e6
        obs_time = Time(ts_total, format="unix", scale="ut1", precision=6)
        dt_iso   = obs_time.iso.replace(" ","T")

        alt = 90.0 - m.th
        az  = (450.0 - m.phi)%360.0

        ra, dec = hor2eq(alt, az, obs_time, location)
        px_val  = 10**(-0.4*m.lsp)
        mag_val = m.lsp + mir_site.offset
        ######## DATA ############################################################################################################
        row = f"{dt_iso},{ra:.6f},{dec:+.6f},{az:.6f},{alt:+.6f},\t" \
              f"{m.cx:.3f},\t{m.cy:.3f},\t{px_val:.0f},\t{mag_val:+.2f}\n"
        ######## DATA ############################################################################################################
        lines.append(row)

    with open(out_file, "a") as f:
        f.writelines(lines)
    print(f"ECSV file saved to: {out_file} with {len(meteors)} meteor frames.")


def process_site(meteors: list, mir_site: MirSite, location: EarthLocation, out_dir: str):
    if not meteors:
        print(f"No meteor data found for site={mir_site.site}.")
        return

    # Derive camera_id from the event or fallback
    camera_id = mir_site.event[-1] if mir_site.event else "C"
    hdr, out_file = generate_ecsv_header(mir_site, camera_id, mir_site.event, out_dir)
    with open(out_file, "w") as f:
        f.write(hdr)

    lines = []
    for m in meteors:
        # Combine integer seconds + microseconds
        ts_total = np.double(m.ts + m.tu / 1e6)  # 'Unix' seconds

        # Build an Astropy Time for UTC
        # scale='utc' => no leap seconds, typical IDL approach close to the 3rd decimal
        obs_time = Time(ts_total, format='unix', scale='utc', precision=6)
        dt_iso   = obs_time.iso.replace(" ","T")

        # alt & az very close to IDL at the 5th decimal
        alt = 90.0 - m.th
        az  = (450.0 - m.phi) % 360.0

        # Use Astropy-based hor2eq_astropy => J2000 coords
        ra_deg, dec_deg = hor2eq_astropy(
            alt_deg=alt,
            az_deg=az,
            obs_time=obs_time,
            location=location,
            pressure_hpa=1013.25,
            temperature_c=15.0
        )

        # Topocentric coordinates from ra_deg, dec_deg to alt, az with no atmospheric refraction
        radec_temp = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='fk5', obstime=obs_time)
        altaz_topo = radec_temp.transform_to(AltAz(obstime=obs_time, location=location, pressure=0))
        # Convert back to alt, az
        alt = altaz_topo.alt.deg
        az = altaz_topo.az.deg

        # Integrated brightness, magnitude
        px_val  = 10 ** (-0.4 * m.lsp)
        mag_val = m.lsp + mir_site.offset

        # Format output row
        dec_str = f"{dec_deg:+.6f}"
        alt_str = f"{alt:+.6f}"
        
        row = (
            f"{dt_iso},"
            f"{ra_deg:.6f},{dec_str},{az:.6f},{alt_str},\t"
            f"{m.cx:.3f},\t{m.cy:.3f},\t{px_val:.0f},\t{mag_val:+.2f}\n"
        )
        lines.append(row)

    with open(out_file, "a") as f:
        f.writelines(lines)

    print(f"ECSV file saved to: {out_file} with {len(meteors)} meteor frames.")

def convert_met_to_ecsv(state_file: str, out_dir: str, photom_dict: dict):

    meteor_list, dir_path, st_arr_np, seq_arr_np,rel_time, ts0, mir_site1, mir_site2 = read_mirfit_state_file_data(state_file)

    # print("\nDEBUG => FINAL site1:", mir_site1)
    # print("DEBUG => FINAL site2:", mir_site2)
    # print(f"Found {len(meteor_list)} total meteors.")

    if not meteor_list:
        print("No meteor data found.")
        return

    site1_mets = [m for m in meteor_list if m.site in [1,4]]
    site2_mets = [m for m in meteor_list if m.site==2]

    mir_site1.offset = mir_site1.offset if 1 not in photom_dict else photom_dict[1]
    mir_site2.offset = mir_site2.offset if 2 not in photom_dict else photom_dict[2]

    loc1 = EarthLocation(lat=mir_site1.lat*u.deg,
                         lon=mir_site1.lon*u.deg,
                         height=mir_site1.elev*u.m)
    loc2 = EarthLocation(lat=mir_site2.lat*u.deg,
                         lon=mir_site2.lon*u.deg,
                         height=mir_site2.elev*u.m)

    process_site(site1_mets, mir_site1, loc1, out_dir)
    process_site(site2_mets, mir_site2, loc2, out_dir)

def parse_photom_arg(photom_str: str):
    """
    Parse a string like "1=15.9,2=16.5" into a dict {1:15.9, 2:16.5, ...}.
    """
    offsets = {}
    if photom_str:
        pairs = photom_str.split(',')
        for pair in pairs:
            pair = pair.strip()
            # each pair is like "1=15.9"
            if '=' in pair:
                site_str, offset_str = pair.split('=')
                site = int(site_str.strip())
                val = float(offset_str.strip())
                offsets[site] = val
    return offsets




if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Translate from .met to .ecsv format.")

    arg_parser.add_argument('input_dir_file', help="Path and file name of the .met file.", default=r"N:\mvovk\Justin_EMCCD\CAP\EMCCD_informed_CAMO_picks\20200725_074335_skyfit2_IAU_DONE\20200725_074335_mir\state_1744244221.met")
    # arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str, help="Path were are store both simulated and observed shower .csv file.")

    arg_parser.add_argument('--photom', help='Comma-separated list of site=offset pairs, by default "1=16.5,2=16.5"', default="1=16.5,2=16.5")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    if not os.path.isfile(cml_args.input_dir_file):
        print('file '+cml_args.input_dir_file+' not found')
        print("You need to specify the correct path and name of of the .met file")
        sys.exit()
    
    # parse the photom offsets:
    photom_dict = parse_photom_arg(cml_args.photom)
    # print photometric offsets for site 1 and 2
    print('Photometric offsets:', photom_dict)

    out_dir = os.path.dirname(cml_args.input_dir_file)
    convert_met_to_ecsv(cml_args.input_dir_file, out_dir, photom_dict)



