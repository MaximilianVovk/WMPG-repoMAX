# Recreate the CSV if missing, fit a 6th-degree log10 polynomial against it,
# and plot CSV ("real") vs polynomial for Earth and Mars from 40–180 km.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime
from nrlmsise00 import msise_model
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, atmDensPoly
import scipy.optimize



def fitAtmPoly_mars(height_min, height_max):
    """ Fits a 7th order polynomial on the atmosphere mass density profile at the given location, time, and 
        for the given height range. From Mars Climate Database CSV data : https://www-mars.lmd.jussieu.fr/mcd_python/
        e.g.  https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py?var1=rho&var2=none&var3=none&var4=none&datekeyhtml=1&ls=160.3&localtime=0.&year=2025&month=10&day=24&hours=18&minutes=41&seconds=6&julian=2460973.278541667&martianyear=38&sol=337&latitude=0&longitude=0&altitude=180000&zkey=3&spacecraft=equator&isfixedlt=off&dust=1&hrkey=1&averaging=loct&dpi=80&islog=off&colorm=jet&minval=&maxval=&proj=cyl&palt=&plon=&plat=&trans=&iswind=off&latpoint=&lonpoint=

    Arguments:
        height_min: [float] Minimum height in meters. E.g. 30000 or 60000 are good values.
        height_max: [float] Maximum height in meters. E.g. 120000 or 180000 are good values.

    Return:
        dens_co: [list] Coeffs for the 7th order polynomial.
    """

    # Generate a height array
    height_arr = np.linspace(height_min, height_max, 200) / 1000.0

    # interpolate the mars csv data to get the atmosphere densities
    mars_csv_path = os.path.join(Path(__file__).parent, "Mars_density_0_180_km__1_km.csv")
    # the first row is the header take the column alt_km	rho_mars_kg_m3
    mars_df = pd.read_csv(mars_csv_path)
    alt_km = mars_df['alt_km'].values
    rho_mars_csv = mars_df['rho_mars_kg_m3'].values

    ### sanity check: remove NaN values ###

    m = np.isfinite(alt_km) & np.isfinite(rho_mars_csv)
    alt_km = alt_km[m]
    rho_mars_csv = rho_mars_csv[m]

    order = np.argsort(alt_km)
    alt_km = alt_km[order]
    rho_mars_csv = rho_mars_csv[order]

    # Guard: if CSV doesn't cover the whole range, clip to overlap to avoid extrapolation
    x_min, x_max = alt_km.min(), alt_km.max()
    if height_arr[0] < x_min or height_arr[-1] > x_max:
        # Warn: trimming to data-supported range
        height_arr = height_arr[(height_arr >= x_min) & (height_arr <= x_max)]

    ### interpolation ###

    atm_densities = np.interp(height_arr, alt_km, rho_mars_csv)
    atm_densities_log = np.log10(atm_densities)
    height_arr = height_arr * 1000.0  # back to meters

    # logic function for curve fitting
    def atmDensPolyLog(height_arr, *dens_co):
        return np.log10(atmDensPoly(height_arr, dens_co))

    # Fit the 7th order polynomial
    dens_co, _ = scipy.optimize.curve_fit(atmDensPolyLog, height_arr, atm_densities_log, \
        p0=np.zeros(7), maxfev=10000)

    return dens_co


if __name__ == "__main__":
    # find the folder whaere this script is located
    script_folder = Path(__file__).parent

    # create a for loop that prints a different altitude=180000 from 180000 to 0 with a step of 10000
    # read the csv file Mars_density_0_180_km__1_km.csv in script_folder
    mars_csv_path = os.path.join(script_folder, "Mars_density_0_180_km__1_km.csv")
    # the first row is the header take the column alt_km	rho_mars_kg_m3
    mars_df = pd.read_csv(mars_csv_path)
    alt_km = mars_df['alt_km'].values
    rho_mars_csv = mars_df['rho_mars_kg_m3'].values

    # take only the values from alt_km that are between 40 km to 180 km
    rho_mars_csv = rho_mars_csv[(alt_km >= 40) & (alt_km <= 180)]
    alt_km = alt_km[(alt_km >= 40) & (alt_km <= 180)]

    # fit the polynomial
    dens_co_mars = fitAtmPoly_mars(40*1000, 180*1000)
    # print the coeffs
    print('dens_co_mars:', dens_co_mars)

    # res = msise_model(datetime(2009, 6, 21, 8, 3, 20), 400, 60, -70, 150, 150, 4, lst=16)
    # print(res[0][5]*1000)
    dens_co = fitAtmPoly(0, 0, 40*1000, 180*1000, 2458870.849678207189)
    print('dens_co:', dens_co)
    altitude = range(40, 181)
    density = []
    rho_poly = []
    rho_poly_mars = []
    # find the values for msise_model from 40 to 180 km
    for alt in altitude:
        res = msise_model(datetime(2009, 6, 21, 8, 3, 20), alt, 60, -70, 150, 150, 4, lst=16)
        density.append(res[0][5]*1000)
        rho_poly.append((atmDensPoly(alt*1000, dens_co))) # *1.25
        # print(f"Altitude: {alt} km, \nMSISE Density: {res[0][5]*1000} kg/m^3, \nPoly Density: {(atmDensPoly(alt*1000, dens_co))} kg/m^3")
        # print(f"Altitude: {alt} km, Density: {res[0][5]*1000} kg/m^3")
        rho_poly_mars.append((atmDensPoly(alt*1000, dens_co_mars)))


    # Plots
    plt.figure(figsize=(7,5))
    plt.plot(density, altitude, '-', label="Earth NRLMSIS-00", color='C0', linewidth=2)
    # plt.semilogx(rho_e_csv, alt_km, '-.', label="Earth CSV", color='C0')
    plt.plot(rho_poly, altitude, ':', label="Earth poly fit 7th Poly", color='blue')
    plt.plot(rho_mars_csv, alt_km, '-', label="Mars MCD", color='C1', linewidth=2)
    plt.plot(rho_poly_mars, altitude, ':', label="Mars poly fit 7th Poly", color='red')
    # make the x axis logarithmic
    plt.xscale("log")
    plt.ylabel("Altitude (km)")
    plt.xlabel("Density (kg/m³)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    # join
    plt.savefig( os.path.join(script_folder,"earth_poly_vs_csv_40to180.png"), dpi=150)
    # plt.show()
    plt.close()
