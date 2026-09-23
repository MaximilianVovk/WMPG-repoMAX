import math

import numpy as np


def keplerian_to_state_vectors(a_m, e, i_rad, peri_rad, node_rad, nu_rad, mu_sun):

    """

    Converts Keplerian orbital elements to 3D state vectors (position and velocity)

    in the heliocentric ecliptic frame.


    Args:

        a_m (float): Semi-major axis (meters)

        e (float): Eccentricity

        i_rad (float): Inclination (radians)

        peri_rad (float): Argument of periapsis (radians)

        node_rad (float): Longitude of ascending node (radians)

        nu_rad (float): True anomaly (radians)

        mu_sun (float): Standard gravitational parameter of the Sun


    Returns:

        (np.array, np.array): r_vec (position), v_vec (velocity)

    """

    

    # 1. Calculate distance (r) and semi-latus rectum (p)

    p = a_m * (1 - e**2)

    r = p / (1 + e * math.cos(nu_rad))


    # 2. Calculate position and velocity in the perifocal (orbital) frame (P, Q, W)

    # P-axis points to periapsis, Q-axis is 90 deg in direction of motion

    r_perifocal = np.array([r * math.cos(nu_rad), r * math.sin(nu_rad), 0])

    

    v_perifocal = np.array([

        -math.sqrt(mu_sun / p) * math.sin(nu_rad),

        math.sqrt(mu_sun / p) * (e + math.cos(nu_rad)),

        0

    ])


    # 3. Create 3D rotation matrix from perifocal to ecliptic (X, Y, Z) frame

    # This is a 3-1-3 Euler angle rotation

    cos_peri = math.cos(peri_rad)

    sin_peri = math.sin(peri_rad)

    cos_node = math.cos(node_rad)

    sin_node = math.sin(node_rad)

    cos_i = math.cos(i_rad)

    sin_i = math.sin(i_rad)


    # Rotation matrix

    R = np.array([

        [cos_node * cos_peri - sin_node * sin_peri * cos_i, -cos_node * sin_peri - sin_node * cos_peri * cos_i,  sin_node * sin_i],

        [sin_node * cos_peri + cos_node * sin_peri * cos_i, -sin_node * sin_peri + cos_node * cos_peri * cos_i, -cos_node * sin_i],

        [sin_peri * sin_i                                ,  cos_peri * sin_i                                ,  cos_i          ]

    ])


    # 4. Apply rotation

    r_vec = R.dot(r_perifocal)

    v_vec = R.dot(v_perifocal)


    return r_vec, v_vec


def get_planet_velocity_vector(r_vec, mu_sun):

    """

    Calculates a planet's velocity vector at a given position,

    assuming a circular, prograde orbit in the ecliptic plane (i=0).

    """

    r_mag = np.linalg.norm(r_vec)

    v_mag = math.sqrt(mu_sun / r_mag)

    

    # Ecliptic north pole

    z_axis = np.array([0, 0, 1])

    

    # Position vector projected onto the ecliptic plane

    r_ecliptic_plane = np.array([r_vec[0], r_vec[1], 0])

    r_ecliptic_norm = np.linalg.norm(r_ecliptic_plane)

    

    if r_ecliptic_norm == 0:

        # This would only happen at a 90-degree inclination,

        # but good to have a fallback.

        return np.array([0, v_mag, 0])


    # Velocity vector is perpendicular to position (tangential)

    # (z_axis x r_vec_normalized) gives the direction of prograde motion

    v_dir = np.cross(z_axis, r_ecliptic_plane / r_ecliptic_norm)

    return v_dir * v_mag


def calculate_atmospheric_entry_speed(v_inf_mag, mu_planet, r_planet, h_atmos):

    """

    Calculates the final atmospheric entry speed using v_infinity.

    """

    r_entry = r_planet + h_atmos

    v_escape_at_h_sq = (2 * mu_planet / r_entry)

    v_i_sq = v_inf_mag**2 + v_escape_at_h_sq

    return math.sqrt(v_i_sq), math.sqrt(v_escape_at_h_sq)


def calculate_3d_intercept_speeds(a_au, e, i_deg, peri_deg, node_deg, verbose=False):

    """

    Calculates the 3D intercept speeds for Earth and Mars using full

    Keplerian orbital elements.

    """

    

    # --- 1. Define Astronomical Constants (SI Units: m, kg, s) ---

    MU_SUN = 1.32712440018e20  # m^3/s^2

    AU_M = 1.495978707e11      # m

    H_ATMOS = 100e3           # m (100 km)


    # Planet data: [mu, radius, orbit_radius]

    planets = {

        "Earth": [3.986004418e14, 6371e3, 1.0 * AU_M],

        "Mars":  [4.282837e13,    3389.5e3, 1.524 * AU_M]

    }


    # --- 2. Convert Object's Orbit to SI and Radians ---

    a_m = a_au * AU_M

    i_rad = math.radians(i_deg)

    peri_rad = math.radians(peri_deg)

    node_rad = math.radians(node_deg)

    
    if verbose:

        print("="*60)

        print(f"Calculating 3D Intercept Speeds for Object:")

        print(f"  a = {a_au} AU, e = {e}, i = {i_deg} deg")

        print(f"  peri = {peri_deg} deg, node = {node_deg} deg")

        print("="*60)


        # --- 3. Earth Intercept Calculation (Unambiguous) ---

        print("\n--- AT EARTH (1.000 AU) ---")

    p_m = a_m * (1 - e**2)
    f_asc = -peri_rad
    f_desc = math.pi - peri_rad
    r_asc = p_m / (1 + e * math.cos(f_asc))
    r_desc= p_m / (1 + e * math.cos(f_desc))


    target_r = planets["Earth"][2]
    f_node = f_asc if abs(r_asc - target_r) < abs(r_desc - target_r) else f_desc
    r_at_node = p_m / (1 + e * math.cos(f_node))

    if verbose:
        print(f"Orbit distance at chosen node: {r_at_node / AU_M:.4f} AU")

        

        if not math.isclose(r_at_node, planets["Earth"][2], rel_tol=0.01):

            print("! WARNING: This orbit does not intersect Earth's orbit (1 AU)")

            print("!          at a node. The provided elements are for a")

            print("!          near-miss or a different orbit than presumed.")

            # We will proceed anyway, assuming intercept is at 1.0 AU

            # and at the true anomaly closest to the node.                  

    

    # Calculate state vectors for the object at intercept

    r_obj_earth, v_obj_earth = keplerian_to_state_vectors(

        a_m, e, i_rad, peri_rad, node_rad, f_node, MU_SUN

    )

    

    # Get Earth's velocity vector at that same position

    v_planet_earth = get_planet_velocity_vector(r_obj_earth, MU_SUN)

    

    # Calculate relative velocity (v-infinity)

    v_inf_vec_earth = v_obj_earth - v_planet_earth

    v_inf_mag_earth = np.linalg.norm(v_inf_vec_earth)

    

    # Calculate final entry speed

    v_i_earth, v_esc_earth = calculate_atmospheric_entry_speed(

        v_inf_mag_earth, planets["Earth"][0], planets["Earth"][1], H_ATMOS

    )

    if verbose:
        print(f"1. Object Heliocentric Vel (v_hel):\t {np.linalg.norm(v_obj_earth) / 1000:.3f} km/s")

        print(f"2. Earth Heliocentric Vel (v_earth):\t {np.linalg.norm(v_planet_earth) / 1000:.3f} km/s")

        print(f"3. Relative Velocity (v_inf):\t\t {v_inf_mag_earth / 1000:.3f} km/s")

        print(f"   (Speed from gravity well:\t\t {v_esc_earth / 1000:.3f} km/s)")

        print(f"4. Final Entry Speed (v_i(Earth)):\t {v_i_earth / 1000:.3f} km/s")




    # --- 4. Mars Intercept Calculation (Two Possibilities) ---
    if verbose:
        print("\n\n--- AT MARS (1.524 AU) ---")

    r_intersect_mars = planets["Mars"][2]

    

    # Solve for true anomaly (nu) where r = r_intersect_mars

    # r = p / (1 + e*cos(nu)) => cos(nu) = (p/r - 1) / e

    cos_nu_mars = (p_m / r_intersect_mars - 1) / e

    if verbose:

        print(f"Orbit distance at chosen node: {r_intersect_mars / AU_M:.4f} AU")

        if abs(cos_nu_mars) > 1:

            print("! ERROR: This orbit never reaches 1.524 AU.")

            print("         Cannot calculate Mars intercept.")

            print("="*60)

            return


    nu_mars_rad = math.acos(max(-1.0, min(1.0, cos_nu_mars)))

    

    # We now have two solutions: +nu (outbound) and -nu (inbound)

    scenarios = {

        "Outbound Intercept": nu_mars_rad,

        "Inbound Intercept": -nu_mars_rad

    }

    v_obj_mars_arr = []
    v_inf_mag_mars_arr = []
    v_i_mars_arr = []

    for name, nu_rad in scenarios.items():
        if verbose:
            print(f"\n  --- Scenario: {name} (nu = {math.degrees(nu_rad):.3f} deg) ---")

        

        # Calculate state vectors for the object at this nu

        r_obj_mars, v_obj_mars = keplerian_to_state_vectors(

            a_m, e, i_rad, peri_rad, node_rad, nu_rad, MU_SUN

        )

        

        # Get Mars's velocity vector at that same position (assuming circular, i=0)

        v_planet_mars = get_planet_velocity_vector(r_obj_mars, MU_SUN)

        

        # Calculate relative velocity (v-infinity)

        v_inf_vec_mars = v_obj_mars - v_planet_mars

        v_inf_mag_mars = np.linalg.norm(v_inf_vec_mars)

        

        # Calculate final entry speed

        v_i_mars, v_esc_mars = calculate_atmospheric_entry_speed(

            v_inf_mag_mars, planets["Mars"][0], planets["Mars"][1], H_ATMOS

        )

        if verbose:
            print(f"  1. Object Heliocentric Vel (v_hel):\t {np.linalg.norm(v_obj_mars) / 1000:.3f} km/s")

            print(f"  2. Mars Heliocentric Vel (v_mars):\t {np.linalg.norm(v_planet_mars) / 1000:.3f} km/s")

            print(f"  3. Relative Velocity (v_inf):\t\t {v_inf_mag_mars / 1000:.3f} km/s")

            print(f"     (Speed from gravity well:\t\t {v_esc_mars / 1000:.3f} km/s)")

            print(f"  4. Final Entry Speed (v_i(Mars)):\t {v_i_mars / 1000:.3f} km/s")

        v_obj_mars_arr.append(np.linalg.norm(v_obj_mars) / 1000)
        v_inf_mag_mars_arr.append(v_inf_mag_mars / 1000)
        v_i_mars_arr.append(v_i_mars / 1000)

    if verbose:
        print("="*60)

    return (np.array(v_obj_mars_arr),
            np.array(v_inf_mag_mars_arr),
            np.array(v_i_mars_arr),
            np.linalg.norm(v_obj_earth)/1e3,
            v_inf_mag_earth/1e3,
            v_i_earth/1e3)




if __name__ == "__main__":

    

    # --- Example Orbit from your request ---

    a_au = 2.101281

    e = 0.542881

    i_deg = 1.267608

    peri_deg = 208.184897

    node_deg = 183.500678

    
    a_au = 7.348131
    e = 0.911266
    i_deg = 171.704358
    peri_deg = 73.032179
    node_deg = 80.812641

    a_au = 546.214833 
    e = 0.998373
    i_deg = 166.331125 
    peri_deg = 219.336283 
    node_deg = 185.554975 

    # Run the full 3D calculation

    _ = calculate_3d_intercept_speeds(a_au, e, i_deg, peri_deg, node_deg, verbose=True)