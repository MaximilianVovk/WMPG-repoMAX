"""
The script will parse the LCAM FTPdectectinfo file, outputting a list detection start and end times.
Use the LCAM detection times to create a folder for each detection in the detection file that contains PNGs 
for every frame between the start and end time (plus a small buffer) in the corresponding EMCCD vid file with Vidchop. 
1.	Create the detection file parsing script first. RMS codebase called readFTPdetectinfo in the file RMS/Formats/FTPdetectinfo.py. 
2.	For each detection, write the detection_num, filename, time of first measurement, and time of last measurement to disk 
latest update: 2024-03-12
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import RMS
import os
import matplotlib.dates as mdates
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from RMS.Formats.FFfile import filenameToDatetime
from datetime import datetime, timedelta
import math
from scipy.interpolate import UnivariateSpline
import shutil
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.signal import correlate

def parse_LCAM(LCAM_name, input_dir_LCAM, input_dir_EMCCD, output_dir, camera_name = "IMAX678-8mm", EMCCD_name = "EMCCD", satellite_name = "", method_diff = 'time'):
    # get current directory

    fps_LCAM = 25 # LCAM fps
    # fps = 32.0 # EMCCD fps

    # split input_path in directory and file name
    input_path_LCAM, file_name_LCAM = os.path.split(input_dir_LCAM)

    # read the EMCCD file
    input_path_EMCCD, file_name_EMCCD = os.path.split(input_dir_EMCCD)

    if file_name_EMCCD[-4:] == 'ecsv':
        # read the EMCCD file skip the lines with #
        df_EMCCD = pd.read_csv(input_dir_EMCCD, delimiter=',', header=0, comment='#')

        # in the column datetime convert the string to datetime
        df_EMCCD['datetime'] = pd.to_datetime(df_EMCCD['datetime'], format='%Y-%m-%d %H:%M:%S.%f')

        # delete column x_image  y_image  integrated_pixel_value
        # df_EMCCD = df_EMCCD.drop(columns=['x_image', 'y_image', 'integrated_pixel_value'])
        # check if the column integrated_pixel_value exists
        if 'integrated_pixel_value' in df_EMCCD.columns:
            # delete column integrated_pixel_value
            df_EMCCD = df_EMCCD.drop(columns=['integrated_pixel_value'])
            

    elif file_name_EMCCD[-3:] == 'csv': # Satellite	Date	JD	Alt	Az	RA	Dec # datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value,mag_data
        # read the csv file
        df_EMCCD = pd.read_csv(input_dir_EMCCD, delimiter=',', header=0)
        if satellite_name == "":
            print("Error: Please provide a satellite name to filter the EMCCD data")
            return
        # make df_EMCCD pd only have the Satellite that have satellite_name but satellite_name can have a space
        # df_EMCCD = df_EMCCD[df_EMCCD['Satellite'].str.contains(satellite_name)]
        df_EMCCD = df_EMCCD[df_EMCCD['Satellite'] == satellite_name]
        # delete the index
        df_EMCCD = df_EMCCD.reset_index(drop=True)
        # change the column Satellite to datetime and Alt to altitude and Az to azimuth
        df_EMCCD = df_EMCCD.rename(columns={'Date':'datetime', 'Alt':'altitude', 'Az':'azimuth'})
        # in the column datetime convert the string to datetime 10/06/2024  06:48:56
        df_EMCCD['datetime'] = pd.to_datetime(df_EMCCD['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        # add a column mag_data with the value 0
        df_EMCCD['mag_data'] = 0

    else:
        # print error message
        print(f"Error: {file_name_EMCCD} is not a valid file format")

    df_EMCCD_OG = df_EMCCD.copy()
    # if input_dir_LCAM has a ecsv instead of a txt file
    if file_name_LCAM[-4:] == 'ecsv':
        # read the ecsv skip the lines with #
        df_LCAM = pd.read_csv(input_dir_LCAM, delimiter=',', header=0, comment='#')
        # in the column datetime convert the string to datetime
        df_LCAM['datetime'] = pd.to_datetime(df_LCAM['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        # delete column x_image  y_image  integrated_pixel_value
        # df_LCAM = df_LCAM.drop(columns=['x_image', 'y_image', 'integrated_pixel_value'])
        df_LCAM = df_LCAM.drop(columns=['integrated_pixel_value'])

        if EMCCD_name == "Dave's Ephemeris":
            # run the interpolator
            # deep copy df_LCAM in original
            # df_LCAM_original = df_LCAM.copy()
            df_EMCCD = shorter_ephemeris(df_LCAM, df_EMCCD)
            # implemet the fit to find th best time offset
            time_offset, df_LCAM_opt = time_offset_to_match(df_LCAM, df_EMCCD)
            print('resampled',EMCCD_name,'data for the same time and optimize the time offset to match the',camera_name,'data')
    
    elif file_name_LCAM[-3:] == 'csv': # Satellite	Date	JD	Alt	Az	RA	Dec # datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value,mag_data
        # read the csv file
        df_LCAM = pd.read_csv(input_dir_LCAM, delimiter=',', header=0)
        if satellite_name == "":
            print("Error: Please provide a satellite name to filter the LCAM data")
            return
        # make df_LCAM pd only have the Satellite that have satellite_name
        # df_LCAM = df_LCAM[df_LCAM['Satellite'].str.contains(satellite_name)]
        df_LCAM = df_LCAM[df_LCAM['Satellite'] == satellite_name]
        # delete the index
        df_LCAM = df_LCAM.reset_index(drop=True)
        # change the column Satellite to datetime and Alt to altitude and Az to azimuth
        df_LCAM = df_LCAM.rename(columns={'Date':'datetime', 'Alt':'altitude', 'Az':'azimuth'})
        # in the column datetime convert the string to datetime 10/06/2024  06:48:56
        df_LCAM['datetime'] = pd.to_datetime(df_LCAM['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        # add a column mag_data with the value 0
        df_LCAM['mag_data'] = 0

    else:

        # def readFTPdetectinfo(ff_directory, file_name, ret_input_format=False):
        event_list = readFTPdetectinfo(input_path_LCAM, file_name_LCAM)
        
        # create a panda dataframe with the columns datetime          ra        dec    azimuth   altitude  mag_data
        df_LCAM = pd.DataFrame(columns=['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'mag_data'])
        ii_LCAM=0
        LCAM_name = LCAM_name.split(',')
        for filename in LCAM_name:
            LCAM_name_curr = filename.strip()
            for n_detect_event in range(len(event_list)):

                if event_list[n_detect_event][0] == LCAM_name_curr:
                    print(f"Detection {n_detect_event} found in the {camera_name} file")

                    detection_num = event_list[n_detect_event][3]
                    # filename = event_list[n_detect_event][0]
                    FF_name = str(event_list[n_detect_event][0])
                    datetime_file = filenameToDatetime(FF_name)

                    # multiply the time of the detection by the fps to get the frame number
                    frame_list=[event_list[n_detect_event][11][i][1]*(1/fps_LCAM) for i in range(len(event_list[n_detect_event][11]))]
                    # from the frame number list get the time of the detection
                    ii_frame_LCAM=0
                    for frame in frame_list:
                        time_measurement = timedelta(seconds=frame)
                        time_measurement = datetime_file + time_measurement
                        time_measurement = time_measurement.strftime('%Y-%m-%d %H:%M:%S.%f')
                        time_measurement = pd.to_datetime(time_measurement, format='%Y-%m-%d %H:%M:%S.%f')

                        # save time_measurement in a list df_json.loc[len(df_json)] = 
                        df_LCAM.loc[ii_LCAM] = [time_measurement,\
                                                event_list[n_detect_event][11][ii_frame_LCAM][4],\
                                                event_list[n_detect_event][11][ii_frame_LCAM][5],\
                                                event_list[n_detect_event][11][ii_frame_LCAM][6],\
                                                event_list[n_detect_event][11][ii_frame_LCAM][7],\
                                                event_list[n_detect_event][11][ii_frame_LCAM][-1]]
                        ii_LCAM += 1
                        ii_frame_LCAM += 1
    
    df_LCAM_OG = df_LCAM.copy()
    if file_name_EMCCD[-4:] == 'ecsv':
        if camera_name == "Dave's Ephemeris":
            # run the interpolator
            # df_LCAM_original = df_LCAM.copy()
            df_LCAM = shorter_ephemeris(df_EMCCD, df_LCAM)
            # implemet the fit to find th best time offset
            time_offset, df_EMCCD_opt = time_offset_to_match(df_EMCCD, df_LCAM)
            
            print('resampled',camera_name,'data for the same time and optimize the time offset to match the',EMCCD_name,'data')

    # print(df_LCAM)

    # use np.unwrap(alpha_rad) to get the angle in the range -pi to pi and make it back to degrees
    df_LCAM['azimuth'] = np.rad2deg(np.unwrap(np.deg2rad(df_LCAM['azimuth'])))
    df_LCAM['altitude'] = np.rad2deg(np.unwrap(np.deg2rad(df_LCAM['altitude'])))
    # use np.unwrap(alpha_rad) to get the angle in the range -pi to pi and make it back to degrees
    df_EMCCD['azimuth'] = np.rad2deg(np.unwrap(np.deg2rad(df_EMCCD['azimuth'])))
    df_EMCCD['altitude'] = np.rad2deg(np.unwrap(np.deg2rad(df_EMCCD['altitude'])))
             
    # find the highest and lowest value of the mag_data
    max_mag = df_LCAM['mag_data'].max()
    min_mag = df_LCAM['mag_data'].min()

    if EMCCD_name != "EMCCD":
        df_EMCCD['mag_data']=np.mean(df_LCAM['mag_data'])
    if camera_name != "IMAX678-8mm" and camera_name != "IMAX678-25mm":
        df_LCAM['mag_data']=np.mean(df_EMCCD['mag_data'])
    # for EMCCD data
    max_mag_EMCCD = df_EMCCD['mag_data'].max()
    min_mag_EMCCD = df_EMCCD['mag_data'].min()

    # find the highest and lowest value of all
    max_mag_all = max(max_mag, max_mag_EMCCD)
    min_mag_all = min(min_mag, min_mag_EMCCD)

    moreLCAM_than_EMCCD=False
    duration_LCAM=df_LCAM['datetime'].iloc[-1]-df_LCAM['datetime'].iloc[0]
    duration_EMCCD=df_EMCCD['datetime'].iloc[-1]-df_EMCCD['datetime'].iloc[0]
    # transform in seconds
    duration_LCAM_sec=duration_LCAM.total_seconds()
    duration_EMCCD_sec=duration_EMCCD.total_seconds()
    if duration_LCAM_sec>duration_EMCCD_sec:
        moreLCAM_than_EMCCD=True


###############################################################
    # plot the azimuth  altitude wit mag_data as color
    fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [1, 1.2]})

    cbar = ax[0].scatter(df_LCAM['azimuth'], df_LCAM['altitude'], c=df_LCAM['mag_data'], cmap='viridis_r')
    ax[0].set_xlabel('azimuth [deg]')
    ax[0].set_ylabel('altitude [deg]')
    ax[0].set_title(camera_name+' data')
    # plot the colorbar
    plt.colorbar(cbar, label='Apparent Magnitude [-]')
    # define the colorbar range
    cbar.set_clim(min_mag_all, max_mag_all)
    ax[0].grid()
    # set the color of the background
    ax[0].set_facecolor('xkcd:black')

    # plot the EMCCD data with the same color as the LCAM data
    cbar2=ax[1].scatter(df_EMCCD['azimuth'], df_EMCCD['altitude'], c=df_EMCCD['mag_data'], cmap='viridis_r')
    ax[1].set_xlabel('azimuth [deg]')
    ax[1].set_ylabel('altitude [deg]')
    ax[1].set_title(EMCCD_name+' data')
    cbar2.set_clim(min_mag_all, max_mag_all)
    # put the grid on
    ax[1].grid()
    # set the color of the background
    ax[1].set_facecolor('xkcd:black')
    if moreLCAM_than_EMCCD==False:
        # set the y axis to be the same
        ax[1].set_ylim(ax[0].get_ylim())
        # set the x axis to be the same
        ax[1].set_xlim(ax[0].get_xlim())
    elif moreLCAM_than_EMCCD==True:
        # set the y axis to be the same
        ax[0].set_ylim(ax[1].get_ylim())
        # set the x axis to be the same
        ax[0].set_xlim(ax[1].get_xlim())

    save_y=ax[0].get_ylim()
    save_x=ax[0].get_xlim()

    # maximize the window
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    # save the plot
    plt.savefig(output_dir+'\\'+'azimuth_altitude_'+camera_name+'_'+EMCCD_name+'.png', dpi=300)
    
    # close the plot
    plt.close()

###############################################################
    # print(df_LCAM)
    # print(df_EMCCD)
    skyplot_alt_az(df_LCAM_OG, df_EMCCD_OG, camera_name, EMCCD_name, satellite_name, output_dir)

###############################################################

    if camera_name == "Dave's Ephemeris":
        # put df_LCAM in EMCCD and viceversa
        df_LCAM, df_EMCCD = df_EMCCD, df_LCAM
        df_LCAM_opt = df_EMCCD_opt
        camera_name, EMCCD_name = EMCCD_name, camera_name

            

    # find the closest df_EMCCD['azimuth'] to df_LCAM['azimuth'] and df_EMCCD['altitude'] to df_LCAM['altitude']
    # for every detection in df_LCAM
    
    if EMCCD_name == "Dave's Ephemeris":
        
        interpolated_alt, interpolated_az = interp_data_min([0], df_LCAM, df_EMCCD)
        for ii_LCAM in range(len(df_LCAM)):

            # find use the index with the same datetime to find the closest azimuth and altitude 
            # Extract the time series for interpolation
            # interpolated_alt, interpolated_az = interp_data_min(offset, df_source, df_ephemeris)
            idx_azimuth = df_EMCCD['datetime'].sub(df_LCAM['datetime'][ii_LCAM]).abs().idxmin()
            idx_altitude = df_EMCCD['datetime'].sub(df_LCAM['datetime'][ii_LCAM]).abs().idxmin()

            # print(f"azimuth and altitude are the same for detection {ii_LCAM} in the EMCCD data")
            # save the closest azimuth and altitude in the dataframe as a new column azimuth_error and altitude_error and the difference in magnitude and time  
            df_LCAM.loc[ii_LCAM, 'azimuth_error'] = df_LCAM['azimuth'][ii_LCAM]-interpolated_az[ii_LCAM]
            df_LCAM.loc[ii_LCAM, 'altitude_error'] = df_LCAM['altitude'][ii_LCAM]-interpolated_alt[ii_LCAM]

            df_LCAM_opt.loc[ii_LCAM, 'azimuth_error'] = df_LCAM['azimuth'][ii_LCAM]-df_LCAM_opt['azimuth'][ii_LCAM]
            df_LCAM_opt.loc[ii_LCAM, 'altitude_error'] = df_LCAM['altitude'][ii_LCAM]-df_LCAM_opt['altitude'][ii_LCAM]

            df_LCAM.loc[ii_LCAM, 'mag_error'] = df_LCAM['mag_data'][ii_LCAM]-df_EMCCD['mag_data'][idx_azimuth]
            # time_error in seconds.milliseconds
            df_LCAM.loc[ii_LCAM, 'time_error'] = (df_LCAM['datetime'][ii_LCAM]-df_EMCCD['datetime'][idx_azimuth]).total_seconds()

    
    else:
        for ii_LCAM in range(len(df_LCAM)):

            if method_diff == 'time':
                # find the closest azimuth and altitude
                idx_time = df_EMCCD['datetime'].sub(df_LCAM['datetime'][ii_LCAM]).abs().idxmin()

                # print(f"azimuth and altitude are the same for detection {ii_LCAM} in the EMCCD data")
                # save the closest azimuth and altitude in the dataframe as a new column azimuth_error and altitude_error and the difference in magnitude and time  
                df_LCAM.loc[ii_LCAM, 'azimuth_error'] = df_LCAM['azimuth'][ii_LCAM]-df_EMCCD['azimuth'][idx_time]
                df_LCAM.loc[ii_LCAM, 'altitude_error'] = df_LCAM['altitude'][ii_LCAM]-df_EMCCD['altitude'][idx_time]

                df_LCAM.loc[ii_LCAM, 'mag_error'] = df_LCAM['mag_data'][ii_LCAM]-df_EMCCD['mag_data'][idx_time]
                # time_error in seconds.milliseconds
                df_LCAM.loc[ii_LCAM, 'time_error'] = (df_LCAM['datetime'][ii_LCAM]-df_EMCCD['datetime'][idx_time]).total_seconds()

            elif method_diff == 'angle':
                # find the closest azimuth and altitude
                idx_azimuth = df_EMCCD['azimuth'].sub(df_LCAM['azimuth'][ii_LCAM]).abs().idxmin()
                idx_altitude = df_EMCCD['altitude'].sub(df_LCAM['altitude'][ii_LCAM]).abs().idxmin()

                # print(f"azimuth and altitude are the same for detection {ii_LCAM} in the EMCCD data")
                # save the closest azimuth and altitude in the dataframe as a new column azimuth_error and altitude_error and the difference in magnitude and time  
                df_LCAM.loc[ii_LCAM, 'azimuth_error'] = df_LCAM['azimuth'][ii_LCAM]-df_EMCCD['azimuth'][idx_azimuth]
                df_LCAM.loc[ii_LCAM, 'altitude_error'] = df_LCAM['altitude'][ii_LCAM]-df_EMCCD['altitude'][idx_altitude]

                df_LCAM.loc[ii_LCAM, 'mag_error'] = df_LCAM['mag_data'][ii_LCAM]-df_EMCCD['mag_data'][idx_azimuth]
                # time_error in seconds.milliseconds
                df_LCAM.loc[ii_LCAM, 'time_error'] = (df_LCAM['datetime'][ii_LCAM]-df_EMCCD['datetime'][idx_azimuth]).total_seconds()
            else:
                # raise a warning ask to define the method_diff
                raise ValueError("Error: Please define the method_diff as 'time' or 'angle'")

                
    # print the error in the time with non nan
    # print(df_LCAM['time_error'][df_LCAM['time_error'].notna()])
    
    # plot the lcam data with the error in the azimuth and altitude
    fig, ax = plt.subplots(3,3,gridspec_kw={'width_ratios': [0.5, 3, 0.5],'height_ratios': [1, 3, 1]})
    # delete the first plot
    ax[0,2].axis('off')
    ax[2,0].axis('off')
    ax[2,2].axis('off')
    ax[0,0].axis('off')

    deg_az = np.sqrt(np.mean(df_LCAM['azimuth_error'][df_LCAM['azimuth_error'].notna()]**2))
    deg_alt = np.sqrt(np.mean(df_LCAM['altitude_error'][df_LCAM['altitude_error'].notna()]**2))
    deg_az_mean = np.mean(df_LCAM['azimuth_error'][df_LCAM['azimuth_error'].notna()])
    deg_alt_mean = np.mean(df_LCAM['altitude_error'][df_LCAM['altitude_error'].notna()])

    degrees_az, minutes_az, seconds_az = deg_to_dms(deg_az)
    degrees_alt, minutes_alt, seconds_alt = deg_to_dms(deg_alt)

    # print only 4 digits of time_offset with print(f)

    # create a a title for the plot
    if EMCCD_name == "Dave's Ephemeris" or camera_name == "Dave's Ephemeris":

        deg_az_opt=np.sqrt(np.mean(df_LCAM_opt['azimuth_error']**2))
        deg_alt_opt=np.sqrt(np.mean(df_LCAM_opt['altitude_error']**2))
        deg_az_opt_mean=np.mean(df_LCAM_opt['azimuth_error'])
        deg_alt_opt_mean=np.mean(df_LCAM_opt['altitude_error'])
        degrees_az_opt, minutes_az_opt, seconds_az_opt = deg_to_dms(deg_az_opt)
        degrees_alt_opt, minutes_alt_opt, seconds_alt_opt = deg_to_dms(deg_alt_opt)

        # put also the best time offset
        fig.suptitle("error values base on time = "+camera_name+" data minus data intepolated base on "+EMCCD_name+" data\n \
        Best time offset: "+str(np.round(time_offset,5))+" s\n\
        RMSD azimuth err. "+str(degrees_az)+"° "+str(minutes_az)+"' "+str(np.round(seconds_az,2))+"'' \
        RMSD altitude err. "+str(degrees_alt)+"° "+str(minutes_alt)+"' "+str(np.round(seconds_alt,2))+"''\n\
        RMSD opt azimuth err. "+str(degrees_az_opt)+"° "+str(minutes_az_opt)+"' "+str(np.round(seconds_az_opt,2))+"'' \
        RMSD opt altitude err. "+str(degrees_alt_opt)+"° "+str(minutes_alt_opt)+"' "+str(np.round(seconds_alt_opt,2))+"''")

        ax[1,1].plot(df_EMCCD['azimuth'], df_EMCCD['altitude'],'-' ,color='r', label=EMCCD_name)
        ax[1,1].plot(interpolated_az, interpolated_alt, '.',color='C0', label='Camera data')
        ax[1,1].plot(df_LCAM_opt['azimuth'], df_LCAM_opt['altitude'], '.',color='C1', label='Best time Offset data')
        ax[1,1].plot(df_LCAM['azimuth'], df_LCAM['altitude'], 'x',color='C0', label='Camera data interp')
        # x for 
        # mpthy circle
        # lines
        # remove from df_LCAM_opt the last row
        df_LCAM_opt = df_LCAM_opt[:-1]

        # plot in [0,0] two dots with the same color as the plot and delete the plot and left only the legend
        ax[0,0].plot(0,0,'-', color='r', label=EMCCD_name)
        ax[0,0].plot(0,0, '.', color='C0', label='No Time Offset')
        ax[0,0].plot(0,0, '.', color='C1', label='Best Time Offset')
        ax[0,0].plot(0,0, 'x', label='Real Data')
        ax[0,0].legend(loc='center left', bbox_to_anchor=(-1.2, 0.5), facecolor='white', framealpha=1)

    else:
        fig.suptitle("error values base on alt & az = "+camera_name+" data minus "+EMCCD_name+" data\n \
        RMSD time err. "+str(np.round(np.sqrt(np.mean(df_LCAM['time_error'][df_LCAM['time_error'].notna()]**2)),4))+" s \
        RMSD magnitude err. "+str(np.round(np.sqrt(np.mean(df_LCAM['mag_error'][df_LCAM['mag_error'].notna()]**2)),4))+" mag\n\
        RMSD azimuth err. "+str(degrees_az)+"° "+str(minutes_az)+"' "+str(np.round(seconds_az,2))+"'' \
        RMSD altitude err. "+str(degrees_alt)+"° "+str(minutes_alt)+"' "+str(np.round(seconds_alt,2))+"''")

        # plot the azimuth  altitude wit mag_data as color
        ax[1,1].plot(df_EMCCD['azimuth'], df_EMCCD['altitude'], '.',color='r')  
        ax[1,1].plot(df_LCAM['azimuth'], df_LCAM['altitude'], '.',color='b')           

    ax[1,1].set_xlabel('azimuth [deg]')
    ax[1,1].set_ylabel('altitude [deg]')
    
    # ax[1,1].set_title('LCAM data')
    # plot the colorbar
    # plt.colorbar(cbar, label='Apparent Magnitude [-]')
    # define the colorbar range
    ax[1,1].grid()
    # set the color of the background
    ax[1,1].set_facecolor('xkcd:black')

    ax[1,1].set_xlim(save_x)
    ax[1,1].set_ylim(save_y)

###############################################################

    # on ax[1,1] the sides of the plot put the error in the azimuth and altitude as a value with one axis
    ax[2,1].bar(df_LCAM['azimuth'], df_LCAM['azimuth_error'], width=abs(ax[1,1].get_xlim()[1]-ax[1,1].get_xlim()[0])/200)
    # ax[2,1].scater(df_LCAM['azimuth'], df_LCAM['azimuth_error'])
    ax[2,1].set_ylabel('az.err.[deg]')
    # set the same x axis as the plot above
    ax[2,1].set_xlim(ax[1,1].get_xlim())
    # place the x axis along the zero
    ax[2,1].spines['bottom'].set_position(('data', 0))
    # place the ticks along the zero
    ax[2,1].xaxis.set_ticks_position('bottom')
    # delete the numbers from the x axis
    ax[2,1].xaxis.set_tick_params(labelbottom=False)
    # delte the border of the plot
    ax[2,1].spines['right'].set_color('none')
    ax[2,1].spines['top'].set_color('none')
    # make the same y axis positive and negative
    ax[2,1].set_ylim(np.max(abs(df_LCAM['azimuth_error']))*-1,np.max(abs(df_LCAM['azimuth_error'])))
    # ticks = ax[2,1].get_yticks()
    # ticks = [tick for tick in ticks if tick != 0]
    # ax[2,1].set_yticks(ticks)
    ax[2,1].axhline(y=deg_az_mean, color='C0', linestyle='--')
    # add the txt of the value of the line at the end of it
    ax[2,1].text(ax[1,1].get_xlim()[-1], deg_az_mean, str(np.round(deg_az_mean,4)), color='C0')
    

    # on ax[1,1] the sides of the plot put the error in the altitude as a value with one axis, with the same number of column as the data 
    ax[1,0].barh(df_LCAM['altitude'],df_LCAM['altitude_error'], height=abs(ax[1,1].get_ylim()[1]-ax[1,1].get_ylim()[0])/100)
    ax[1,0].set_xlabel('alt.err.[deg]')
    # set the same y axis as the plot above
    ax[1,0].set_ylim(ax[1,1].get_ylim())
    # place the ticks along the zero
    ax[1,0].yaxis.set_ticks_position('left')
    # delete the numbers from the y axis
    ax[1,0].yaxis.set_tick_params(labelleft=False)
    # delte the border of the plot
    ax[1,0].spines['right'].set_color('none')
    ax[1,0].spines['top'].set_color('none')
    ax[1,0].spines['left'].set_position(('data', 0))
    # make the same y axis positive and negative
    ax[1,0].set_xlim(np.max(abs(df_LCAM['altitude_error']))*-1,np.max(abs(df_LCAM['altitude_error'])))
    # roatate the x axis ticks 45 degrees
    ax[1,0].tick_params(axis='x', rotation=45)
    # do a vertical line with np.mean(df_LCAM['altitude_error'][df_LCAM['altitude_error'].notna()])
    ax[1,0].axvline(x=deg_alt_mean, color='C0', linestyle='--')
    ax[1,0].text(deg_alt_mean,ax[1,1].get_ylim()[-1], str(np.round(deg_alt_mean,4)), color='C0')
    

    # ticks = ax[1,0].get_xticks()
    # ticks = [tick for tick in ticks if tick != 0]
    # ax[1,0].set_xticks(ticks)
    if EMCCD_name == "Dave's Ephemeris" or camera_name == "Dave's Ephemeris":
        # plot in ax[1,2] the df_LCAM_opt['altitude_error']
        ax[1,2].barh(df_LCAM_opt['altitude'],df_LCAM_opt['altitude_error'], height=abs(ax[1,1].get_ylim()[1]-ax[1,1].get_ylim()[0])/100, color='C1')
        ax[1,2].set_xlabel('alt.opt.err.[deg]')
        # set the same y axis as the plot above
        ax[1,2].set_ylim(ax[1,1].get_ylim())
        # place the y axis along the zero
        ax[1,2].spines['left'].set_position(('data', 0))
        # place the ticks along the zero
        ax[1,2].yaxis.set_ticks_position('left')
        # delete the numbers from the y axis
        ax[1,2].yaxis.set_tick_params(labelleft=False)
        ax[1,2].spines['right'].set_color('none')
        ax[1,2].spines['top'].set_color('none')
        ax[1,2].spines['left'].set_position(('data', 0))
        # roatate the x axis ticks 45 degrees
        ax[1,2].tick_params(axis='x', rotation=45)
        ax[1,2].set_xlim(np.max(abs(df_LCAM_opt['altitude_error']))*-1,np.max(abs(df_LCAM_opt['altitude_error'])))
        ax[1,2].axvline(x=deg_alt_opt_mean, color='C1', linestyle='--')
        ax[1,2].text(deg_alt_opt_mean, ax[1,1].get_ylim()[-1], str(np.round(deg_alt_opt_mean,4)), color='C1')
        

    else:
        # on ax[1,2] the sides of the plot put the error in the magnitude as a value with one axis
        ax[1,2].scatter(df_LCAM['mag_error'],df_LCAM['altitude'], color='turquoise')
        ax[1,2].set_xlabel('mag.err.[deg]')
        # set the same y axis as the plot above
        ax[1,2].set_ylim(ax[1,1].get_ylim())
        # place the y axis along the zero
        ax[1,2].spines['left'].set_position(('data', 0))
        # place the ticks along the zero
        ax[1,2].yaxis.set_ticks_position('left')
        # delete the numbers from the y axis
        ax[1,2].yaxis.set_tick_params(labelleft=False)
        # invert the y axis
        ax[1,2].invert_xaxis()
        # delte the border of the plot
        ax[1,2].spines['right'].set_color('none')
        ax[1,2].spines['top'].set_color('none')
        ax[1,2].axvline(x=np.mean(df_LCAM['mag_error'][df_LCAM['mag_error'].notna()]), color='turquoise', linestyle='--')
        ax[1,2].text(np.mean(df_LCAM['mag_error'][df_LCAM['mag_error'].notna()]), ax[1,1].get_ylim()[-1], str(np.round(np.mean(df_LCAM['mag_error'][df_LCAM['mag_error'].notna()]),4)), color='turquoise')
        
    if EMCCD_name == "Dave's Ephemeris" or camera_name == "Dave's Ephemeris":
        # plot in ax[0,1] the df_LCAM_opt['azimuth_error']
        ax[0,1].bar(df_LCAM_opt['azimuth'], df_LCAM_opt['azimuth_error'], color='C1', width=abs(ax[1,1].get_xlim()[1]-ax[1,1].get_xlim()[0])/200)
        ax[0,1].set_ylabel('az.opt.err.[deg]')
        # set the same x axis as the plot above
        ax[0,1].set_xlim(ax[1,1].get_xlim())
        # place the x axis along the zero
        ax[0,1].spines['bottom'].set_position(('data', 0))
        # place the ticks along the zero
        ax[0,1].xaxis.set_ticks_position('bottom')
        # delete the numbers from the x axis
        ax[0,1].xaxis.set_tick_params(labelbottom=False)
        # delte the border of the plot
        ax[0,1].spines['right'].set_color('none')
        ax[0,1].spines['top'].set_color('none')
        # make the same y axis positive and negative
        ax[0,1].set_ylim(np.max(abs(df_LCAM_opt['azimuth_error']))*-1,np.max(abs(df_LCAM_opt['azimuth_error'])))
        ax[0,1].axhline(y=deg_az_opt_mean, color='C1', linestyle='--')
        ax[0,1].text(ax[1,1].get_xlim()[-1], deg_az_opt_mean, str(np.round(deg_az_opt_mean,4)), color='C1')

    else:
        ax[0,1].scatter(df_LCAM['azimuth'], df_LCAM['time_error'], color='teal')
        ax[0,1].set_ylabel('time.err.[s]')
        # set the same x axis as the plot above
        ax[0,1].set_xlim(ax[1,1].get_xlim())
        # place the x axis along the zero
        ax[0,1].spines['bottom'].set_position(('data', 0))
        # place the ticks along the zero
        ax[0,1].xaxis.set_ticks_position('bottom')
        # delete the numbers from the x axis
        ax[0,1].xaxis.set_tick_params(labelbottom=False)
        # delte the border of the plot
        ax[0,1].spines['right'].set_color('none')
        ax[0,1].spines['top'].set_color('none')
        # make the same y axis positive and negative
        # ax[0,1].set_ylim(np.max(abs(df_LCAM['time_error']))*-1,np.max(abs(df_LCAM['time_error'])))
        ax[0,1].axhline(y=np.mean(df_LCAM['time_error'][df_LCAM['time_error'].notna()]), color='teal', linestyle='--')
        ax[0,1].text(ax[1,1].get_xlim()[-1], np.mean(df_LCAM['time_error'][df_LCAM['time_error'].notna()]), str(np.round(np.mean(df_LCAM['time_error'][df_LCAM['time_error'].notna()]),4)), color='teal')
        # ticks = ax[0,1].get_yticks()
        # ticks = [tick for tick in ticks if tick != 0]
        # ax[0,1].set_yticks(ticks)                 

###############################################################

    # create a list called alt_az_positions_LCAM that is a list of tuples (altitude, azimuth)
    alt_az_positions_LCAM = [(df_LCAM['altitude'][i], df_LCAM['azimuth'][i]) for i in range(len(df_LCAM))]
    # take only the first and last value of the alt_az_positions_LCAM
    alt_az_positions_LCAM = [alt_az_positions_LCAM[0], alt_az_positions_LCAM[-1]]

    # give the values of df_LCAM['datetime'] in seconds as a list of numbers
    timestamp_s_LCAM = df_LCAM['datetime'].view('int64') / 1000000000
    timestamp_s_LCAM_first_last=timestamp_s_LCAM.tolist()
    # take only the first and last value of the timestamp_s_LCAM
    timestamp_s_LCAM_first_last = [timestamp_s_LCAM_first_last[0], timestamp_s_LCAM_first_last[-1]]

    # calculate the average angular velocity of the satellite in degrees per second
    avg_deg_per_sec_LCAM = calculate_deg_per_second(alt_az_positions_LCAM, timestamp_s_LCAM_first_last)
    # print the average angular velocity and average pixel displacement per frame
    print(f"Average angular velocity ({camera_name}): {avg_deg_per_sec_LCAM} degrees per second")
    avg_displacement_LCAM=0
    if 'x_image' in df_LCAM.columns and 'y_image' in df_LCAM.columns:
        # create a list pixel_positions_LCAM is a list of tuples (x, y) coordinates of the satellite in each frame
        pixel_positions_LCAM = [(df_LCAM['x_image'][i], df_LCAM['y_image'][i]) for i in range(len(df_LCAM))]
        # # take only the first and last value of the pixel_positions_LCAM
        pixel_positions_LCAM_first_last = [pixel_positions_LCAM[0], pixel_positions_LCAM[-1]]
        # calculate the average pixel displacement per frame
        displacements_per_frame_LCAM, avg_displacement_LCAM = pixel_displacement_per_frame(pixel_positions_LCAM_first_last, len(pixel_positions_LCAM))
        # displacements_per_frame_LCAM, avg_displacement_LCAM = pixel_displacement_per_frame(pixel_positions_LCAM)
        print(f"Average pixel displacement per frame ({camera_name}): {avg_displacement_LCAM} pixels")


    # create a list called alt_az_positions_EMCCD that is a list of tuples (altitude, azimuth)
    alt_az_positions_EMCCD = [(df_EMCCD['altitude'][i], df_EMCCD['azimuth'][i]) for i in range(len(df_EMCCD))]
    # take only the first and last value of the alt_az_positions_EMCCD
    alt_az_positions_EMCCD = [alt_az_positions_EMCCD[0], alt_az_positions_EMCCD[-1]]

    timestamp_s_EMCCD = df_EMCCD['datetime'].view('int64') / 1000000000
    timestamp_s_EMCCD_first_last=timestamp_s_EMCCD.tolist()
    # take only the first and last value of the timestamp_s_EMCCD
    timestamp_s_EMCCD_first_last = [timestamp_s_EMCCD_first_last[0], timestamp_s_EMCCD_first_last[-1]]

    # calculate the average angular velocity of the satellite in degrees per second
    avg_deg_per_sec_EMCCD = calculate_deg_per_second(alt_az_positions_EMCCD, timestamp_s_EMCCD_first_last)

    # print the average angular velocity and average pixel displacement per frame
    print(f"Average angular velocity ({EMCCD_name}): {avg_deg_per_sec_EMCCD} degrees per second")

    avg_displacement_EMCCD=0
    # check x_image and y_image exists
    if 'x_image' in df_EMCCD.columns and 'y_image' in df_EMCCD.columns:
        # create a list pixel_positions_EMCCD is a list of tuples (x, y) coordinates of the satellite in each frame
        pixel_positions_EMCCD = [(df_EMCCD['x_image'][i], df_EMCCD['y_image'][i]) for i in range(len(df_EMCCD))]
        # # take only the first and last value of the pixel_positions_EMCCD
        pixel_positions_EMCCD_first_last = [pixel_positions_EMCCD[0], pixel_positions_EMCCD[-1]]
        # calculate the average pixel displacement per frame
        displacements_per_frame_EMCCD, avg_displacement_EMCCD = pixel_displacement_per_frame(pixel_positions_EMCCD_first_last, len(pixel_positions_EMCCD))
        # displacements_per_frame_EMCCD, avg_displacement_EMCCD = pixel_displacement_per_frame(pixel_positions_EMCCD)
        print(f"Average pixel displacement per frame ({EMCCD_name}): {avg_displacement_EMCCD} pixels")



###############################################################


    # give more space between the plots
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # make it full screen
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    # save the plot
    plt.savefig(output_dir+'\\'+'error_'+camera_name+'_'+EMCCD_name+'.png', dpi=300)

    plt.close()


    # subplot for the mag and time of EMCCD and LCAM
    # fig, ax = plt.subplots(2,2,gridspec_kw={'width_ratios': [1, 1],'height_ratios': [1, 1]})
    # create a a title for the plot
    # fig.suptitle(f"EMCCD data minus LCAM data, errors plot only when azimuth and altitude are the same")

    # add a subplot for the mag difference
    fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
    # flat the ax
    ax = ax.flatten()

    # plot the mag of the LCAM and EMCCD
    ax[0].scatter(df_EMCCD['datetime'], df_EMCCD['mag_data'], c='r', label=EMCCD_name)
    # subtract the first value of the timestamp_s_EMCCD to make it start from zero
    spline_EMCCD = UnivariateSpline(timestamp_s_EMCCD, df_EMCCD['mag_data'], s=100)
    ax[0].plot(df_EMCCD['datetime'], spline_EMCCD(timestamp_s_EMCCD), c='r')

    # check if there are enought data for spline function to plot the spline
    # if len(df_EMCCD['datetime'])>10:
    #     # get the spline of the average base on the spline of the mag_data
    #     ax[0].plot(df_EMCCD['datetime'], df_EMCCD['mag_data'].rolling(window=10, center=True).mean(), c='r')

    ax[0].scatter(df_LCAM['datetime'], df_LCAM['mag_data'], c='b', label=camera_name)
    # use from scipy.interpolate the UnivariateSpline
    spline_LCAM = UnivariateSpline(timestamp_s_LCAM, df_LCAM['mag_data'], s=100)
    ax[0].plot(df_LCAM['datetime'], spline_LCAM(timestamp_s_LCAM), c='b')

    # check if there are enought data for spline function to plot the spline
    # if len(df_LCAM['datetime'])>10:
    #     # get the spline of the average base on the spline of the mag_data
    #     ax[0].plot(df_LCAM['datetime'], df_LCAM['mag_data'].rolling(window=10, center=True).mean(), c='b')

    # on the ticks also show the hours
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))

    ax[0].set_xlabel('Time HH:MM:SS.ms')
    ax[0].set_ylabel('Apparent Magnitude [-]')
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
    ax[0].legend()
    # grid on
    ax[0].grid()

    if moreLCAM_than_EMCCD==True:
        # x axis base on the LCAM data
        ax[0].set_xlim(np.max(df_EMCCD['datetime']), np.min(df_EMCCD['datetime']))
    elif moreLCAM_than_EMCCD==False:
        # x axis base on the LCAM data
        ax[0].set_xlim(np.max(df_LCAM['datetime']), np.min(df_LCAM['datetime']))

    # ivert the y axis
    ax[0].invert_yaxis()
    ax[0].invert_xaxis()

    ax[1].scatter(df_LCAM['datetime'], df_LCAM['mag_error'], c='k')
    # use the UnivariateSpline  
    spline_LCAM_error = UnivariateSpline(timestamp_s_LCAM, df_LCAM['mag_error'], s=100)
    ax[1].plot(df_LCAM['datetime'], spline_LCAM_error(timestamp_s_LCAM), c='k')
    # if len(df_LCAM['mag_error'])>10:
    #     ax[1].plot(df_LCAM['datetime'], df_LCAM['mag_error'].rolling(window=10, center=True).mean(), c='k')

    ax[1].set_ylabel('mag.res[-]')
    # set the same x axis as the plot above
    ax[1].set_xlim(ax[0].get_xlim())
    # place the x axis along the zero
    ax[1].spines['bottom'].set_position(('data', 0))
    # place the ticks along the zero
    ax[1].xaxis.set_ticks_position('bottom')
    # delete the numbers from the x axis
    ax[1].xaxis.set_tick_params(labelbottom=False)
    # delte the border of the plot
    ax[1].spines['right'].set_color('none')
    ax[1].spines['top'].set_color('none')
    # add more thicks along the y axis
    ax[1].yaxis.set_major_locator(plt.MaxNLocator(5))
    ax[1].grid()

    # put as the suptitle the average difference betweee df_LCAM['mag_data'].rolling(window=10).mean() and df_EMCCD['mag_data'].rolling(window=10).mean() and make it 2 decimals and bold
    fig.suptitle(f"Average mag residual {abs(np.round(np.mean(df_LCAM['mag_error']),2))}")
    # create a subtitle with the average angular velocity and average pixel displacement per frame rounded to 2 decimals
    if avg_displacement_EMCCD == 0 and avg_displacement_LCAM == 0:
        ax[0].set_title(f"avg angular velocity {EMCCD_name} {np.round(avg_deg_per_sec_EMCCD,2)} deg/s,\n\
                        avg angular velocity {camera_name} {np.round(avg_deg_per_sec_LCAM,2)} deg/s")        
    elif avg_displacement_EMCCD == 0:
        ax[0].set_title(f"avg angular velocity {EMCCD_name} {np.round(avg_deg_per_sec_EMCCD,2)} deg/s,\n\
                        avg angular velocity {camera_name} {np.round(avg_deg_per_sec_LCAM,2)} deg/s, avg pixel angular velocity {camera_name} {np.round(avg_displacement_LCAM,2)} pixels/frame")
    elif avg_displacement_LCAM == 0:
        ax[0].set_title(f"avg angular velocity {EMCCD_name} {np.round(avg_deg_per_sec_EMCCD,2)} deg/s, avg pixel angular velocity {EMCCD_name} {np.round(avg_displacement_EMCCD,2)} pixels/frame\n\
                        avg angular velocity {camera_name} {np.round(avg_deg_per_sec_LCAM,2)} deg/s")
    else:
        ax[0].set_title(f"avg angular velocity EMCCD {np.round(avg_deg_per_sec_EMCCD,2)} deg/s, avg pixel angular velocity EMCCD {np.round(avg_displacement_EMCCD,2)} pixels/frame\n\
        avg angular velocity {camera_name} {np.round(avg_deg_per_sec_LCAM,2)} deg/s, avg pixel angular velocity {camera_name} {np.round(avg_displacement_LCAM,2)} pixels/frame")

    # full screen
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    # save the plot
    plt.savefig(output_dir+'\\'+'mag_'+camera_name+'_'+EMCCD_name+'.png', dpi=300)
    
    plt.close()



def find_closest_index(time_arr, time_sampled):
    closest_indices = []
    for sample in time_sampled:
        closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
        closest_indices.append(closest_index)
    return closest_indices


def time_offset_to_match(df_source, df_ephemeris):
    # Initial guess for the time offset
    initial_guess = 0

    print(df_source, df_ephemeris)

    # Use scipy's minimize function to find the optimal time offset
    result = minimize(compute_difference, [initial_guess], args=(df_source, df_ephemeris), method='Nelder-Mead',tol=1e-8)
    
    # Extract the optimal time offset from the result
    optimal_offset = result.x[0]
    
    # Extract the time series for interpolation
    interpolated_alt, interpolated_az = interp_data_min([optimal_offset], df_source, df_ephemeris)

    # Create the interpolated dataframe
    df_source_shifted = pd.DataFrame({
        'datetime': df_source['datetime'] + pd.to_timedelta(optimal_offset, unit='s'),  # Convert back to datetime
        'altitude': interpolated_alt,
        'azimuth': interpolated_az,
        'mag_data': 0  # Set magnitude to zero
    })
    
    return optimal_offset, df_source_shifted


def compute_difference(offset, df_source, df_ephemeris):
    # Extract the time series for interpolation
    interpolated_alt, interpolated_az = interp_data_min(offset, df_source, df_ephemeris)

    # create a loop to calculate the ang_dist for each value of the df_source
    ang_dist_list=[]
    for i in range(len(df_source)):
        ang_dist=angular_distance(df_source['altitude'][i], df_source['azimuth'][i], interpolated_alt[i], interpolated_az[i])
        ang_dist_list.append(ang_dist)

    ssr = np.sum(np.array(ang_dist_list)**2)
    
    print(f"Offset: {offset[0]}, Total difference: {ssr}")

    return ssr


# def interp_data_min_old(offset, df_source, df_ephemeris):
#     # Copy the dataframes to avoid modifying the originals
#     df_source_shifted = df_source.copy()
#     df_source_shifted['datetime'] = df_source_shifted['datetime'] + pd.to_timedelta(offset[0], unit='s')

#     # Extract the time series for interpolation
#     source_times_shifted = df_source_shifted['datetime'].view('int64') / 1e9  # Convert to seconds
#     ephemeris_times = df_ephemeris['datetime'].view('int64') / 1e9  # Convert to seconds

#     ephemeris_times_limited = ephemeris_times[(ephemeris_times-1 >= source_times_shifted.iloc[0]) & (ephemeris_times+1 <= source_times_shifted.iloc[-1])]

#     # find the index of ephemris alt and az for ephemeris_times_limited
#     closest_indices = find_closest_index(ephemeris_times, ephemeris_times_limited)

#     # Interpolation functions for ephemeris data
#     alt_interp = interp1d(ephemeris_times_limited, df_ephemeris['altitude'][closest_indices], kind='cubic', fill_value="extrapolate")
#     az_interp = interp1d(ephemeris_times_limited, df_ephemeris['azimuth'][closest_indices], kind='cubic', fill_value="extrapolate")
#     # Calculate SSR for each time offset
#     interpolated_alt = alt_interp(source_times_shifted)
#     interpolated_az = az_interp(source_times_shifted)

#     return interpolated_alt, interpolated_az

def interp_data_min(offset, df_source, df_ephemeris):
    # Copy the dataframes to avoid modifying the originals
    df_source_shifted = df_source.copy()
    df_source_shifted['datetime'] = df_source_shifted['datetime'] + pd.to_timedelta(offset[0], unit='s')

    # Extract the time series for interpolation
    source_times_shifted = df_source_shifted['datetime'].view('int64') / 1e9  # Convert to seconds
    ephemeris_times = df_ephemeris['datetime'].view('int64') / 1e9  # Convert to seconds

    ephemeris_times_limited = ephemeris_times[(ephemeris_times-1 >= source_times_shifted.iloc[0]) & (ephemeris_times+1 <= source_times_shifted.iloc[-1])]

    # find the index of ephemeris alt and az for ephemeris_times_limited
    closest_indices = find_closest_index(ephemeris_times, ephemeris_times_limited)

    if len(ephemeris_times_limited) < 4 or len(closest_indices) < 4:
        print("Not enough points for cubic interpolation, switching to linear interpolation")
        # Interpolation functions for ephemeris data using linear method
        alt_interp = interp1d(ephemeris_times_limited, df_ephemeris['altitude'][closest_indices], kind='linear', fill_value="extrapolate")
        az_interp = interp1d(ephemeris_times_limited, df_ephemeris['azimuth'][closest_indices], kind='linear', fill_value="extrapolate")
    else:
        # Interpolation functions for ephemeris data using cubic method
        alt_interp = interp1d(ephemeris_times_limited, df_ephemeris['altitude'][closest_indices], kind='cubic', fill_value="extrapolate")
        az_interp = interp1d(ephemeris_times_limited, df_ephemeris['azimuth'][closest_indices], kind='cubic', fill_value="extrapolate")

    # Calculate SSR for each time offset
    interpolated_alt = alt_interp(source_times_shifted)
    interpolated_az = az_interp(source_times_shifted)

    return interpolated_alt, interpolated_az



def interpolate_to_ephemeris_change_number_ephemeris_data(df_source, df_ephemeris, method='cubicSpline'):
    # Ensure both dataframes are sorted by datetime
    df_source = df_source.sort_values(by='datetime').reset_index(drop=True)
    df_ephemeris = df_ephemeris.sort_values(by='datetime').reset_index(drop=True)
    
    # Extract the time series for interpolation
    source_times = df_source['datetime'].view('int64') / 1e9  # Convert to seconds
    ephemeris_times = df_ephemeris['datetime'].view('int64') / 1e9  # Convert to seconds
    ephemeris_times_limited = ephemeris_times[(ephemeris_times+5 >= source_times.iloc[0]) & (ephemeris_times-5 <= source_times.iloc[-1])]

    # find the index of ephemris alt and az for ephemeris_times_limited
    closest_indices = find_closest_index(ephemeris_times, ephemeris_times_limited)

    # print initial lenght of the ephemeris data
    print(f"Initial length of ephemeris data: {len(df_ephemeris)}")
    # print differece between 0 and 1 of time
    print(f"Time difference between Dave's frames: {df_ephemeris['datetime'].iloc[1] - df_ephemeris['datetime'].iloc[0]}")

    # Interpolate altitude
    alt_interp_func_ephem = interp1d(ephemeris_times_limited, df_ephemeris['altitude'][closest_indices], kind='linear', fill_value="extrapolate")

    # Interpolate azimuth
    az_interp_func_ephem = interp1d(ephemeris_times_limited, df_ephemeris['azimuth'][closest_indices], kind='linear', fill_value="extrapolate")

    # now get new values for the altitude and azimuth every every 0.00001 seconds
    ephemeris_times_new = np.arange(ephemeris_times_limited.iloc[0], ephemeris_times_limited.iloc[-1], 0.5)

    # Interpolate altitude
    alt_interp_ephemeris = alt_interp_func_ephem(ephemeris_times_new)

    # Interpolate azimuth
    az_interp_ephemeris = az_interp_func_ephem(ephemeris_times_new)

    # Create the interpolated dataframe
    df_ephemeris_interpolated = pd.DataFrame({
        'datetime': pd.to_datetime(ephemeris_times_new * 1e9),  # Convert back to datetime
        'altitude': alt_interp_ephemeris,
        'azimuth': az_interp_ephemeris,
        'mag_data': 0  # Set magnitude to zero
    })

    # print initial lenght of the ephemeris data
    print(f"New length of ephemeris data: {len(df_ephemeris_interpolated)}")
    # print differece between 0 and 1 of time
    print(f"New Time difference : {df_ephemeris_interpolated['datetime'].iloc[1] - df_ephemeris_interpolated['datetime'].iloc[0]}")

    print(f"Interpolating the data with the same timing of the ephemeris data using {method} method")

    # fid the index that have ephemeris_times_new in ephemeris_times
    closest_indices_ephemeris = find_closest_index(ephemeris_times, ephemeris_times_new)
#################### CUBIC SPLINE INTERPOLATION ####################
    if method == 'cubicSpline':
        # Create cubic spline interpolators for altitude and azimuth
        # spline_alt = CubicSpline(source_times, df_source['altitude'])
        # spline_az = CubicSpline(source_times, df_source['azimuth'])

        spline_alt = UnivariateSpline(ephemeris_times_new, df_ephemeris['altitude'][closest_indices_ephemeris], s=100)
        spline_az = UnivariateSpline(ephemeris_times_new, df_ephemeris['azimuth'][closest_indices_ephemeris], s=100)
        
        # Interpolate the altitude and azimuth at the limited ephemeris times
        alt_interp = spline_alt(source_times)
        az_interp = spline_az(source_times)

#################### POLYNOMIAL INTERPOLATION ###################
    elif method == 'polynomial':
        # Fit polynomial to altitude and azimuth
        poly_alt = np.polyfit(source_times, df_source['altitude'], 4)
        poly_az = np.polyfit(source_times, df_source['azimuth'], 4)

        # Evaluate polynomial at the limited ephemeris times
        alt_interp = np.polyval(poly_alt, ephemeris_times_new)
        az_interp = np.polyval(poly_az, ephemeris_times_new)

#################### LINEAR REGRESSION ####################
    elif method == 'linear':
        # Fit linear regression to altitude and azimuth
        slope_alt, intercept_alt, _, _, _ = linregress(source_times, df_source['altitude'])
        slope_az, intercept_az, _, _, _ = linregress(source_times, df_source['azimuth'])

        # Evaluate linear regression at the limited ephemeris times
        alt_interp = intercept_alt + slope_alt * ephemeris_times_new
        az_interp = intercept_az + slope_az * ephemeris_times_new

#################### INTERPOLATION ####################
    elif method == 'interpolation':

        # Interpolate altitude
        alt_interp_func = interp1d(source_times, df_source['altitude'], kind='linear', fill_value="extrapolate")
        alt_interp = alt_interp_func(ephemeris_times_new)

        # Interpolate azimuth
        az_interp_func = interp1d(source_times, df_source['azimuth'], kind='linear', fill_value="extrapolate")
        az_interp = az_interp_func(ephemeris_times_new)

        # # Interpolate altitude
        # alt_interp_func = interp1d(ephemeris_times, df_ephemeris['altitude'], kind='linear', fill_value="extrapolate")
        # alt_interp = alt_interp_func(source_times)

        # # Interpolate azimuth
        # az_interp_func = interp1d(ephemeris_times, df_ephemeris['azimuth'], kind='linear', fill_value="extrapolate")
        # az_interp = az_interp_func(source_times)

# #################### INTERPOLATION 3D ####################
    elif method == '3Dlinear':
        # Prepare data for 3D interpolation
        points = np.vstack((source_times, df_source['altitude'], df_source['azimuth'])).T

        # Create RBF interpolator for altitude and azimuth
        rbf_interpolator = RBFInterpolator(points[:, 0:1], points[:, 1:], kernel='linear', epsilon=1)

        # Interpolate the altitude and azimuth at the limited ephemeris times
        interp_points = rbf_interpolator(ephemeris_times_new[:, np.newaxis])
        alt_interp, az_interp = interp_points[:, 0], interp_points[:, 1]

########################################################

    # Create the interpolated dataframe
    df_source_interpolated = pd.DataFrame({
        'datetime': pd.to_datetime(source_times * 1e9),  # Convert back to datetime
        'altitude': alt_interp,
        'azimuth': az_interp,
        'mag_data': 0  # Set magnitude to zero
    })

    # Delete the NaN values
    df_source_interpolated = df_source_interpolated.dropna()

    # Reset the index
    df_source_interpolated = df_source_interpolated.reset_index(drop=True)

    # print that the interpolation was successful
    print(f"Interpolation successful using {method} method")

    # save_deg_per_second=[]
    # alt_az_positions = [(df_source_interpolated['altitude'][i], df_source_interpolated['azimuth'][i]) for i in range(len(df_source_interpolated))]
    # # for loop to def calculate_deg_per_second for all altitudes and azimuths giving the time in seconds of and i+1 and i
    # for i in range(len(alt_az_positions) - 1):
    #     alt1, az1 = alt_az_positions[i]
    #     alt2, az2 = alt_az_positions[i + 1]
        
    #     # Calculate angular distance between consecutive positions
    #     angle = angular_distance(alt1, az1, alt2, az2)
    #     ang_deg_s = angle / (ephemeris_times_new[i + 1] - ephemeris_times_new[i])
    #     # print(ang_deg_s)
    #     save_deg_per_second.append(ang_deg_s)

    # save_deg_per_second_ephem=[]
    # # print the average angular velocity and average pixel displacement per frame
    # alt_az_positions = [(df_ephemeris_interpolated['altitude'][i], df_ephemeris_interpolated['azimuth'][i]) for i in range(len(df_ephemeris_interpolated))]
    # # for loop to def calculate_deg_per_second for all altitudes and azimuths giving the time in seconds of and i+1 and i
    # for i in range(len(alt_az_positions) - 1):
    #     alt1, az1 = alt_az_positions[i]
    #     alt2, az2 = alt_az_positions[i + 1]
        
    #     # Calculate angular distance between consecutive positions
    #     angle = angular_distance(alt1, az1, alt2, az2)
    #     ang_deg_s = angle / (ephemeris_times_new[i + 1] - ephemeris_times_new[i])
    #     save_deg_per_second_ephem.append(ang_deg_s)
        
    # # print the difference between the average angular velocity of the source and the ephemeris
    # print(f"Average angular velocity of the source: {np.mean(save_deg_per_second)} degrees per second")
    # print(f"Average angular velocity of the ephemeris: {np.mean(save_deg_per_second_ephem)} degrees per second")
    # print(f"Difference between the two: {np.mean(save_deg_per_second)-np.mean(save_deg_per_second_ephem)} degrees per second")
    # print()

    return df_source_interpolated,df_ephemeris_interpolated


def shorter_ephemeris(df_source, df_ephemeris):
    # Ensure both dataframes are sorted by datetime
    df_source = df_source.sort_values(by='datetime').reset_index(drop=True)
    df_ephemeris = df_ephemeris.sort_values(by='datetime').reset_index(drop=True)
    
    # Extract the time series for interpolation
    source_times = df_source['datetime'].view('int64') / 1e9  # Convert to seconds
    ephemeris_times = df_ephemeris['datetime'].view('int64') / 1e9  # Convert to seconds
    ephemeris_times_limited = ephemeris_times[(ephemeris_times+5 >= source_times.iloc[0]) & (ephemeris_times-5 <= source_times.iloc[-1])]
    
    # find the index of ephemris alt and az for ephemeris_times_limited
    closest_indices = ephemeris_times_limited.index

    # Create the interpolated dataframe
    df_ephemeris_interpolated = pd.DataFrame({
        'datetime': pd.to_datetime(ephemeris_times_limited * 1e9),  # Convert back to datetime
        'altitude': df_ephemeris['altitude'][closest_indices],
        'azimuth': df_ephemeris['azimuth'][closest_indices],
        'mag_data': 0  # Set magnitude to zero
    })

    # reset index
    df_ephemeris_interpolated = df_ephemeris_interpolated.reset_index(drop=True)

    return df_ephemeris_interpolated



def skyplot_alt_az(df_LCAM, df_EMCCD, camera_name, EMCCD_name, sat_name, output_dir):
    # Sky sphere plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)

    if camera_name == "Dave's Ephemeris":
        # plot the Alt	Az data
        ax.plot(df_LCAM['azimuth']*np.pi/180, 90-df_LCAM['altitude'], 'b.', label=camera_name)
        ax.plot(df_EMCCD['azimuth']*np.pi/180, 90-df_EMCCD['altitude'], 'r.', label=EMCCD_name)
    else:
        # plot the Alt	Az data
        ax.plot(df_EMCCD['azimuth']*np.pi/180, 90-df_EMCCD['altitude'], 'r.', label=EMCCD_name)
        ax.plot(df_LCAM['azimuth']*np.pi/180, 90-df_LCAM['altitude'], 'b.', label=camera_name)

    # check which the two index whos datetime difference in df_EMCCD is bigger than a minute
    # if the difference is bigger than a minute plot the line between the two points

    # put the time value txt only on the first and last value like text(azimuth, altitude, time.strftime('%H:%M:%S'), fontsize=9, ha='right')
    ax.text(df_EMCCD['azimuth'][0]*np.pi/180, 90-df_EMCCD['altitude'][0], df_EMCCD['datetime'][0].strftime('%H:%M:%S'), fontsize=9, ha='right')
    ax.text(df_EMCCD['azimuth'][len(df_EMCCD)-1]*np.pi/180, 90-df_EMCCD['altitude'][len(df_EMCCD)-1], df_EMCCD['datetime'][len(df_EMCCD)-1].strftime('%H:%M:%S'), fontsize=9, ha='left')
    ax.text(df_LCAM['azimuth'][0]*np.pi/180, 90-df_LCAM['altitude'][0], df_LCAM['datetime'][0].strftime('%H:%M:%S'), fontsize=9, ha='right')
    ax.text(df_LCAM['azimuth'][len(df_LCAM)-1]*np.pi/180, 90-df_LCAM['altitude'][len(df_LCAM)-1], df_LCAM['datetime'][len(df_LCAM)-1].strftime('%H:%M:%S'), fontsize=9, ha='left')

    for ii in range(len(df_EMCCD)-1):
        if (df_EMCCD['datetime'][ii+1]-df_EMCCD['datetime'][ii]).total_seconds()>60:
            ax.text(df_EMCCD['azimuth'][ii]*np.pi/180, 90-df_EMCCD['altitude'][ii], df_EMCCD['datetime'][ii].strftime('%H:%M:%S'), fontsize=9, ha='left')
            ax.text(df_EMCCD['azimuth'][ii+1]*np.pi/180, 90-df_EMCCD['altitude'][ii+1], df_EMCCD['datetime'][ii+1].strftime('%H:%M:%S'), fontsize=9, ha='left')

    for ii in range(len(df_LCAM)-1):
        if (df_LCAM['datetime'][ii+1]-df_LCAM['datetime'][ii]).total_seconds()>60:
            ax.text(df_LCAM['azimuth'][ii]*np.pi/180, 90-df_LCAM['altitude'][ii], df_LCAM['datetime'][ii].strftime('%H:%M:%S'), fontsize=9, ha='left')
            ax.text(df_LCAM['azimuth'][ii+1]*np.pi/180, 90-df_LCAM['altitude'][ii+1], df_LCAM['datetime'][ii+1].strftime('%H:%M:%S'), fontsize=9, ha='left')

    # Setting labels
    ax.set_ylim(0, 90)
    ax.set_yticks(range(0, 91, 10))
    ax.set_yticklabels(map(str, range(90, -1, -10)))
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    plt.title(str(sat_name)+' Skysphere of '+str(camera_name)+' and '+str(EMCCD_name))
    plt.legend()
    plt.grid(True)
    # save the plot in the same folder as the script
    plt.savefig(output_dir+'\\'+'skysphere_'+camera_name+'_'+EMCCD_name+'.png')
    plt.close()
    # plt.show()    

def deg_to_dms(deg):
    degrees = int(deg)
    minutes = int((deg - degrees) * 60)
    seconds = (deg - degrees - minutes / 60) * 3600
    return degrees, minutes, seconds

def angular_distance(alt1, az1, alt2, az2):
    # Convert degrees to radians
    alt1, az1, alt2, az2 = map(math.radians, [alt1, az1, alt2, az2])
    
    # # Using the spherical law of cosines
    # angle = math.acos(math.sin(alt1) * math.sin(alt2) + math.cos(alt1) * math.cos(alt2) * math.cos(az1 - az2))
    # return math.degrees(angle)

    # Haversine formula for angular distance on the sphere
    daz = az2 - az1
    dalt = alt2 - alt1
    a = math.sin(dalt/2)**2 + math.cos(alt1) * math.cos(alt2) * math.sin(daz/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return math.degrees(c)

def calculate_deg_per_second(alt_az_positions, times):
    # alt_az_positions is a list of tuples (altitude, azimuth)
    # times is a list of corresponding timestamps in seconds
    
    total_angle = 0
    for i in range(len(alt_az_positions) - 1):
        alt1, az1 = alt_az_positions[i]
        alt2, az2 = alt_az_positions[i + 1]
        
        # Calculate angular distance between consecutive positions
        angle = angular_distance(alt1, az1, alt2, az2)
        total_angle += angle
    
    # Calculate total time elapsed
    total_time = times[-1] - times[0]
    
    # Degrees per second
    deg_per_sec = total_angle / total_time
    return deg_per_sec

def pixel_displacement_per_frame(pixel_positions,frame_num=0):
    # pixel_positions is a list of tuples (x, y) coordinates of the satellite in each frame
    
    total_displacement = 0
    displacements_per_frame = []
    for i in range(len(pixel_positions) - 1):
        x1, y1 = pixel_positions[i]
        x2, y2 = pixel_positions[i + 1]
        
        # Calculate Euclidean distance between consecutive positions
        displacement = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        displacements_per_frame.append(displacement)
        total_displacement += displacement

    # print(f"Total displacement: {total_displacement} pixels")
    
    if frame_num<=0:
        # Average displacement per frame
        average_displacement = total_displacement / (len(pixel_positions))
    else:
        # Average displacement per frame
        average_displacement = total_displacement / frame_num


    
    return displacements_per_frame, average_displacement

if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fom Observation and simulated data weselect the most likely through PCA, run it, and store results to disk.")

    arg_parser.add_argument('--fits_lcam', metavar='FITS_LCAM', type=str, default="FF_CAWE01_20240312_013910_226_0155136.fits",\
        help="File were is stored and the LCAM .txt file with the detections if not in the input path not considered.")

    arg_parser.add_argument('--escv_LCAM', metavar='ESCV_LCAM', type=str, default="",\
        help="Skyfit2 File were is stored and the LCAM .ecsv file with the detections if not specified will walk in th directories.") # 2024-03-12T01_39_10_RMS_CAWE01.ecsv # FTPdetectinfo_CAWE01_20240311_235515_723529.txt
    
    arg_parser.add_argument('--escv_EMCCD', metavar='ESCV_EMCCD', type=str, default="",\
        help="Skyfit2 File were is stored and the EMCCD .ecsv file with the detections if not specified will walk in th directories.")
    
    arg_parser.add_argument('--sat_name', metavar='SAT_NAME', type=str, default="",\
        help="Name of the satellite, if not given by default will read the input directory to figure the satellite name.")
    
    arg_parser.add_argument('--method_diff', metavar='METHOD_DIFF', type=str, default='time',\
        help="Method of difference between LCAM EMCCD to use : time, angle")

    arg_parser.add_argument('--cal_file', metavar='CAL_FILE', type=str, default=r"C:\Users\maxiv\Documents\UWO\Space Situational Awareness DRDC\Cal_sat\CAWE_calibration",\
        help="Path to the calibration file of Dave's satelite ephemeris.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default="",\
        help="Path to the output directory.")
    # C:\Users\maxiv\Documents\UWO\Space Situational Awareness DRDC\8mm\20240613\CAWE07-EGS_20240613_065326
    # C:\Users\maxiv\Documents\UWO\Space Situational Awareness DRDC\8mm\20240612\CAWE07-EGS_20240612_064700
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Space Situational Awareness DRDC\25mm\20240714\CAWEA6-EGS_031737.34",\
        help="Path were is stored and the LCAM .txt file with the detections.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
    if cml_args.output_dir=="":
        cml_args.output_dir=cml_args.input_dir
    escv_LCAM_found=False
    escv_EMCCD_found=False

    if cml_args.sat_name=="":
        # check if in root there is written CRYOSTAT or EGS or JASON
        if "CRYOSTAT" in cml_args.input_dir:
            sat_name="CRYOSAT 2"
        elif "EGS" in cml_args.input_dir:
            sat_name="EGS (AJISAI)"
        elif "JASON" in cml_args.input_dir:
            sat_name="JASON 2"
        elif "LAGEOS" in cml_args.input_dir:
            sat_name="LAGEOS 1"
        elif "LARETS" in cml_args.input_dir:
            sat_name="LARETS"
        elif "STELLA" in cml_args.input_dir:
            sat_name="STELLA"



    if cml_args.escv_LCAM!="" or cml_args.escv_EMCCD!="":
        # walk in the input_dir and inside of each folder and find the file that end with 678MO25.ecsv and 02F.ecsv
        for root, dirs, files in os.walk(cml_args.input_dir):
            for file in files:
                # check first if there is cml_args.escv_LCAM
                if cml_args.escv_LCAM in file:
                    cml_args.escv_LCAM = file
                    # raise a flag that the file was found
                    escv_LCAM_found=True
                if cml_args.escv_EMCCD in file:
                    cml_args.escv_EMCCD = file
                    # raise a flag that the file was found
                    escv_EMCCD_found=True
        if escv_LCAM_found==False:
            print(f"File {cml_args.escv_LCAM} not found")
        if escv_EMCCD_found==False:
            print(f"File {cml_args.escv_EMCCD} not found")


    # if not found the file print file not found and walk in the input_dir and inside of each folder and find the file that end with 678MO25.ecsv and 02F.ecsv
    if escv_LCAM_found==False:
        for root, dirs, files in os.walk(cml_args.input_dir):
            # look for the file that ends with 678MO25.ecsv and copy it to cml_args.input_dir
            for file in files:
                if file.endswith("CAWE01.ecsv") or file.endswith("CAWE07.ecsv"):
                    cml_args.escv_LCAM = file
                    print(f"File {cml_args.escv_LCAM} found")
                    LCAM_name = "IMAX678-8mm"
                    # raise a flag that the file was found
                    escv_LCAM_found=True
                    # if the file is not in the input_dir copy it to the input_dir
                    if root!=cml_args.input_dir:
                        shutil.copy(os.path.join(root, file), cml_args.input_dir)
                    break
                if file.endswith("MO25.ecsv") or file.endswith("CAEUA1.ecsv") or file.endswith("CAEUA2.ecsv") or file.endswith("CAEUA3.ecsv") or file.endswith("CAEUA4.ecsv") or file.endswith("CAEUA5.ecsv") or file.endswith("CAEUA6.ecsv"):
                    cml_args.escv_LCAM = file
                    print(f"File {cml_args.escv_LCAM} found")
                    LCAM_name = "IMAX678-25mm"
                    # raise a flag that the file was found
                    escv_LCAM_found=True
                    # if the file is not in the input_dir copy it to the input_dir
                    if root!=cml_args.input_dir:
                        shutil.copy(os.path.join(root, file), cml_args.input_dir)
                    break
            if escv_LCAM_found==True:
                break

    if escv_EMCCD_found==False:
        for root, dirs, files in os.walk(cml_args.input_dir):
            # look for the file that ends with 02F.ecsv and copy it to cml_args.input_dir
            for file in files:
                if file.endswith("02F.ecsv"):
                    cml_args.escv_EMCCD = file
                    print(f"File {cml_args.escv_EMCCD} found")
                    # raise a flag that the file was found
                    escv_EMCCD_found=True
                    EMCCD_name = "EMCCD"
                    # if the file is not in the input_dir copy it to the input_dir
                    if root!=cml_args.input_dir:
                        shutil.copy(os.path.join(root, file), cml_args.input_dir)
                    break
            if escv_EMCCD_found==True:
                break


    if escv_LCAM_found==False and escv_EMCCD_found==True: # camera_name = "IMAX678-8mm", EMCCD_name = "EMCCD"
        escv_LCAM_found=True
        LCAM_name = "Dave's Ephemeris"
        # read the ecsv skip the lines with #
        df_EMCCD_date = pd.read_csv(os.path.join(cml_args.input_dir, cml_args.escv_EMCCD), delimiter=',', header=0, comment='#')
        # from the datetime column extract the first value and take the first half before T
        Date = df_EMCCD_date['datetime'][0].split('T')[0]
        # from date delite the '-' and substitute it with ''
        Date = Date.replace('-', '')
        # from in the cal_file read find if any of the files has the date in it
        for root, dirs, files in os.walk(cml_args.cal_file):
            for file in files:
                if Date in file:
                    cml_args.escv_LCAM = file
                    print(f"File {cml_args.escv_LCAM} found")
                    # copy the file to the input_dir
                    shutil.copy(os.path.join(root, file), cml_args.input_dir)
                    break

    elif escv_LCAM_found==True and escv_EMCCD_found==False:
        escv_EMCCD_found=True
        EMCCD_name = "Dave's Ephemeris"
        # read the ecsv skip the lines with #
        df_LCAM_date = pd.read_csv(os.path.join(cml_args.input_dir, cml_args.escv_LCAM), delimiter=',', header=0, comment='#')
        # from the datetime column extract the first value and take the first half before T
        Date = df_LCAM_date['datetime'][0].split('T')[0]
        # from date delite the '-' and substitute it with ''
        Date = Date.replace('-', '')
        # from in the cal_file read find if any of the files has the date in it
        for root, dirs, files in os.walk(cml_args.cal_file):
            for file in files:
                if Date in file:
                    cml_args.escv_EMCCD = file
                    print(f"File {cml_args.escv_EMCCD} found")
                    # copy the file to the input_dir
                    shutil.copy(os.path.join(root, file), cml_args.input_dir)
                    break

    # # join the input_dir with the file name
    input_dir_LCAM = os.path.join(cml_args.input_dir, cml_args.escv_LCAM)
    print(input_dir_LCAM)
    input_dir_EMCCD = os.path.join(cml_args.input_dir, cml_args.escv_EMCCD)
    print(input_dir_EMCCD)

    # create the folder if it does not exist
    if not os.path.exists(cml_args.output_dir):
        os.makedirs(cml_args.output_dir)

    parse_LCAM(cml_args.fits_lcam, input_dir_LCAM, input_dir_EMCCD, cml_args.output_dir,LCAM_name, EMCCD_name, sat_name, cml_args.method_diff)