""" Automatic collecting CAMO narrowfiled .vid files, automatic generating calibration files. """

from __future__ import print_function, division, absolute_import

import os
import re
import sys
import argparse
import errno
import glob
import shutil
import subprocess
import datetime
import matplotlib.pyplot as plt
import bz2
import numpy as np
from MirfitPrepare_recalibrate_plates import processEvents


# Version number
__version__ = 1.3

# Local working directory
work_dir = os.getcwd()


# Data path variables
data_path = '/srv/meteor/klingon/events/'
working_dir_suffix = '_mir'

# Data file parameters
file_prefix = 'ev_'
extensions_list = ['.vid.bz2', '.txt']
station_id_list = ['A_01T', 'A_02T']

# Plates generation parameters
plates_workdir_prefix = 'plates_'
plates_remote_dir = '/srv/meteor/klingon/mirchk/'
plates_reductions = '/srv/meteor/reductions/klingon/plates_man/'
plates_reductions_auto = '/srv/meteor/reductions/klingon/plates_auto/'
plates_default_dir = 'plates_default'
plates_star_regex = '*star*.txt'
plates_type = [['scale', '.aff', 3.0], ['exact', '.ast', 10]] # Last parameter per entry = Max threshold
sites_names = [['Tavistock', '01'], ['Elginfield', '02']]

# Take all 11 colums in mircal results
mircal_parse_column_indices = range(11)

# Row of UNIX time in the meteor data file header
METEOR_HEADER_UNIX_ROW = 5

# Encoder indices in the meteor data file
METEOR_Hx_INDEX = 1
METEOR_Hy_INDEX = 2

# Centroid intides in the stars data file
STAR_Cx_INDEX = 13
STAR_Cy_INDEX = 14

# Encoder indices in the stars data file
STAR_Hx_INDEX = 9
STAR_Hy_INDEX = 10
STAR_Hu_INDEX = 15
STAR_Hv_INDEX = 16

# Lables of individual parameters in the stars files
star_lables = ['Ts', 'tu', 'M', 'th', 'phi', 'rx', 'ry', 'wx', 'wy', 'hx', 'hy', 'nx', 'ny', 'cx', 'cy', 'hu', 'hv', 'z', 'A', 'b', 'c', 'SAO', 'name']

# Formatting of the stars files
star_formats = ['{:010.0f}', '{:06.0f}', '{:03.2f}', '{:07.4f}', '{:+9.4f}', '{:6.2f}', '{:6.2f}', 
    '{:6.2f}', '{:6.2f}', '{:8.2f}', '{:8.2f}', '{:6.2f}', '{:6.2f}', '{:6.2f}', '{:6.2f}', '{:7.2f}', 
    '{:7.2f}', '{:5.2f}', '{:4.0f}', '{:4.1f}', '{:3.2f}', '{:06.0f}', '{:s}']


class AffPlate(object):
    """ AFF type plate structure. """

    def __init__(self):

        self.magic = 0
        self.info_len = 0

        self.r0 = 0
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

        self.sx = 0
        self.sy = 0
        self.phi = 0
        self.tx=0
        self.ty = 0
        self.wid = 0
        self.ht = 0
        self.site = 0
        self.text = ''
        self.text_size = 256
        self.flags = 0


def loadScale(dir_path, file_name):
    """ Loads an AFF scale plate. """

    # Open the file for binary reading
    fid = open(os.path.join(dir_path, file_name), 'rb')

    # Init the plate struct
    scale = AffPlate()


    # Load file info
    scale.magic = int(np.fromfile(fid, dtype=np.uint32, count = 1))
    scale.info_len = int(np.fromfile(fid, dtype=np.uint32, count = 1))

  
    # Load reserved members
    scale.r0 = np.fromfile(fid, dtype=np.uint32, count = 1)
    scale.r1 = np.fromfile(fid, dtype=np.uint32, count = 1)
    scale.r2 = np.fromfile(fid, dtype=np.uint32, count = 1)
    scale.r3 = np.fromfile(fid, dtype=np.uint32, count = 1)

    # Load scaling terms
    scale.sx = np.fromfile(fid, dtype=np.float64, count = 1)
    scale.sy = np.fromfile(fid, dtype=np.float64, count = 1)

    # Load the rotation term
    scale.phi = np.fromfile(fid, dtype=np.float64, count = 1)[0]

    # Load the translation terms
    scale.tx = np.fromfile(fid, dtype=np.float64, count = 1)
    scale.ty = np.fromfile(fid, dtype=np.float64, count = 1)[0]

    # Load the image size parameters
    scale.wid = np.fromfile(fid, dtype=np.int32, count = 1)[0]
    scale.ht = np.fromfile(fid, dtype=np.int32, count = 1)[0]

    # Load the site number
    scale.site = np.fromfile(fid, dtype=np.uint32, count = 1)[0]

    # Load the descriptive comment
    scale.text = np.fromfile(fid, dtype='|S'+str(scale.text_size), count = 1)[0]

    # Load the flags
    scale.flags = np.fromfile(fid, dtype=np.uint32, count = 1)[0]
    print(scale)
    return scale



def plateScaleMap(scale, x, y, reverse_map=False):
    """ Map the image delta coordinates to encoder delta coordinates. """

    # Init the M matrix
    M = np.zeros((3,3))

    M[0,0] =  scale.sx*np.cos(scale.phi)
    M[0,1] = -scale.sy*np.sin(scale.phi)
    M[0,2] =  scale.tx*M[0,0] + scale.ty*M[0,1]

    M[1,0] =  scale.sx*np.sin(scale.phi)
    M[1,1] =  scale.sy*np.cos(scale.phi)
    M[1,2] =  scale.tx*M[1,0] + scale.ty*M[1,1]

    M[2,0] = 0
    M[2,1] = 0
    M[2,2] = 1

    # Run if doing the reverse mapping
    if reverse_map:

        # Calculate the reverse map matrix
        R = np.linalg.inv(M)

        # Reverse mapping
        px = R[0,0]*x + R[0,1]*y + R[0,2]
        py = R[1,0]*x + R[1,1]*y + R[1,2]

        return px, py


    else:

        # Forward mapping
        mx = M[0,0]*x + M[0,1]*y + M[0,2]
        my = M[1,0]*x + M[1,1]*y + M[1,2]

        return mx, my


def parse_star_line(line, star_lables):
    """
    Parse a single line of key-value pairs, skipping unknown labels 
    like narr.beg/corr.end. Return a dict of recognized fields.
    """

    tokens = line.strip().split()
    star_dict = {}

    i = 0
    n = len(tokens)

    while i < n - 1:
        label = tokens[i]
        value_str = tokens[i + 1]

        # ---------------------------
        # EXCEPTION: if label == 'ts', rename it to 'Ts'
        # ---------------------------
        if label == 'ts':
            label = 'Ts'
        # ^ That ensures 'ts' in file is stored under 'Ts' for the star_dict
        # so reorder_star_dict knows to place it in the 'Ts' position.

        # If the label is recognized in star_lables
        if label in star_lables:
            # Special case for 'name' => string, everything else => float
            if label == 'name':
                star_dict[label] = value_str.strip("'")
            else:
                try:
                    star_dict[label] = float(value_str)
                except ValueError:
                    star_dict[label] = 0.0

            i += 2
        else:
            # It's not recognized (maybe narr.beg, corr.end, etc.)
            # Typically label + numeric => skip 2 tokens
            # If the next token isn't numeric, skip 1
            import re
            if re.match(r'^-?\d+(\.\d+)?$', value_str):
                i += 2
            else:
                i += 1

    return star_dict


def reorder_star_dict(star_dict, star_lables):
    """
    Convert the parsed dictionary into a list in exactly the same
    order as star_lables. Missing fields become 0.0 or 'NoName' for 'name'.
    """
    star_line = []
    for lbl in star_lables:
        val = star_dict.get(lbl, None)
        if val is None:
            if lbl == 'name':
                val = 'NoName'
            else:
                val = 0.0
        star_line.append(val)
    return star_line


def loadStarsData(dir_path, file_name, star_lables=star_lables):
    """
    Loads a star file, returns (header, stars_list).
    - If file is missing => returns (False, False).
    - If lines start with 'T' => returns (None, None) => "already corrected".
    - Otherwise parse recognized fields into star_lables order.

    This function references the global star_lables in your script 
    and reuses parse_star_line / reorder_star_dict above.
    """

    file_path = os.path.join(dir_path, file_name)
    if not os.path.isfile(file_path):
        print("File not found:", file_path)
        return False, False

    header = []
    stars_data = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            # Keep header lines
            if line.startswith('#'):
                header.append(line + "\n")
                continue

            # If line starts with 'T' => you used to do "return None, None"
            if line.startswith('T'):
                # means "already corrected"
                return None, None

            # Parse recognized label-value pairs
            star_dict = parse_star_line(line, star_lables)
            if star_dict:
                # reorder them into the same indexing you had before
                star_line = reorder_star_dict(star_dict, star_lables)
                stars_data.append(star_line)

    return header, stars_data



def saveStarsData(dir_path, file_name, header, stars_data):

    # Add 'corr' to the star file name
    file_name = '.'.join(file_name.split('.')[:-1]) + '_corr.txt'

    with open(os.path.join(dir_path, file_name), 'w') as f:

        for line in header:
            f.write(line)
        
        for star in stars_data:

            star_string = ''

            # Generate the output star line in the proper format
            for i, param_format in enumerate(star_formats):

                # Generate a star string
                star_string += star_lables[i] + ' ' + param_format.format(star[i]) + ' '


            # Write star string to file
            f.write(star_string + "\n")



def correctStarsFile(dir_path, site_name, scale_plate_name):
    """ Uses the scale plate to 'virtually' move all stars that are not in the center of corr image, and saves
        the corrected stars file.

        This procedure is ues for obtaining exact plates with a higher precision. 
    """

    for stars_file in os.listdir(dir_path):

        if (site_name in stars_file) and ('star' in stars_file) and not ('corr' in stars_file):

            # If it is a scale pattern file, just rename it to corrected file
            if 'scale_pattern' in stars_file:

                shutil.move(os.path.join(dir_path, stars_file), os.path.join(dir_path, stars_file.replace('.txt', '')+'_corr.txt'))

                continue

                
            # Load the scale plate to memory
            scale = loadScale(dir_path, scale_plate_name)

            # Load the stars file
            header, stars = loadStarsData(dir_path, stars_file)

            # Check if the star file is present or it is already processed
            if stars:

                # Go through every star and correct the encoder positions
                for i, star in enumerate(stars):
                    
                    # Get the 'narr' image centroids
                    cx, cy = star[STAR_Cx_INDEX], star[STAR_Cy_INDEX]

                    # Calculate the offset from the centre
                    dx = cx - scale.wid/2
                    dy = scale.ht/2 - cy

                    # Convert the offset to encoder delta
                    hu, hv = plateScaleMap(scale, dx, dy)

                    # print('d:', dx, dy)
                    # print('h:', hu, hv)

                    # Apply the offset to the absolute encoder coordinates
                    hx = star[STAR_Hx_INDEX] + hu
                    hy = star[STAR_Hy_INDEX] + hv

                    # print('hx:', hx, 'hy:', hy)

                    # Update the stars list
                    stars[i][STAR_Hx_INDEX] = hx
                    stars[i][STAR_Hy_INDEX] = hy
                    star[STAR_Cx_INDEX] = scale.wid/2
                    star[STAR_Cy_INDEX] = scale.ht/2

                print(stars)
                
                # Save modified stars data
                saveStarsData(dir_path, stars_file, header, stars)

                print('Saved: ' + stars_file)


            elif stars is None:
                print('This star file has already been corrected for the star offset!')

            elif stars == False:
                print(stars_file + ' - File not found!')



def printSeparator():
    """ Prints a separator on the screen. """

    print('##################################################################################################')


def mkdirP(path):
    """ Make a directory, check if it exists. """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def formatEventNames(event_list):
    """ Checks the event name format for all events in the list and formats it to YYYYMMDD_hhmmss. """

    def _properFormat(name):
        """ Checks if the event name is of the proper format. """

        if len(name) == 15:
            if name.count('_') == 1:
                name = name.split('_')
                try:
                    date = int(name[0])
                    time = int(name[1])

                    return True

                except:
                    pass

        return False

    formatted_events = []
    for raw_event in event_list:

        # Strip extra spaces at the head and tails of the event name string
        event = raw_event.strip()

        # Cut the name if it is too long, i.e. contains more letters than it should
        if len(event) > 17:
            event = event[:17]

        # Check the appropriate format
        if ':' in event:
            # Web format, convert to normal format

            date = event[:8]
            time = event[8:].strip().split(':')

            event = date + '_' + time[0] + time[1] + time[2]

        # Check if the event has a proper format
        if _properFormat(event):

            # Add the event to the final list
            formatted_events.append(event)

        else:
            print('!!!Event with the name', raw_event, 'could not be processed due to a non standard name!')
            print('   Try using the YYYYMMDD_hhmmss format')

    return formatted_events


def extractNights(event_list):
    """ Extracts night names from the list of events. """

    night_list = []

    for event in event_list:
        night = event.split('_')[0]

        if not (night in night_list):
            night_list.append(night)

    return night_list



def findCalibrationFiles(calib_dir, plates_type, sites_names):
    """ Find calibration files in the given directory. """

    calib_files_list = []

    for type_name, type_extension, _ in plates_type:

            # Run for all sites
            for site, site_no in sites_names:

                output_file_name = type_name + '_' + site_no + type_extension

                out_file_path = os.path.join(calib_dir, output_file_name)

                # Check if the file exists and add it to the list if it does
                if os.path.isfile(out_file_path):
                    calib_files_list.append(output_file_name)


    return calib_files_list



def regexCopy(source, destination):
    """ Copy all the files which match the source regex to destination. """

    for file_path in glob.glob(source):
        
        # Get only the name of the file, not the whole path
        _, file_name = os.path.split(file_path)

        # Copy the file to destionation
        shutil.copy2(file_path, os.path.join(destination, file_name))



def copyStarFiles(server_dir, night_name, plates_local_dir):
    """ Copy all *stars*.txt files from a given night to a local dir. """

    # Make a new local dir for the plates
    plates_local_dir = os.path.join(work_dir, plates_workdir_prefix+night_name)
    mkdirP(plates_local_dir)

    # Copy all the *star*.txt files in the working directory
    regexCopy(os.path.join(server_dir, night_name, plates_star_regex), plates_local_dir)



def parseMirCalResults(results, column_indices):
    """ Parses the Mircal output into a machine readable list. """

    parsed_results = []

    # Split the results by newline
    results = results.decode().split('\n')

    # Take only the index and the resuduals line
    for line in results[1:]:
        line = line.strip().replace('|', '').split()

        # Break if the list is empty
        if not line:
            break

        # Break the loop after the end of good stars list
        if '...' in line[0]:
            break

        if line[0] != '':

            line_list = []

            # Parse column values by the given column index list
            for col_idx in column_indices:
                line_list.append(float(line[col_idx]))

            # Add parsed columns to the line results list
            parsed_results.append(line_list)


    return parsed_results



def runMirCal(site, site_no, type_name, type_extension, bad_list=[], output_command=False, star_corr=False):
    """ Runs mircal and automatically creates plates.
    Args:
        site: [str] site name, e.g. Tavistock
        site_no: [str] site number, e.g. '01'
        type_name: [str] type of the plate to be generated, e.g. 'scale'
        type_extension: [str] file extension for the given plate type, e.g. for scale is '.aff'

    Return:
        
    """

    # Generate the string for stars with high residuals
    bad_star_string = ''
    if bad_list:
        bad_star_string = '--bad '+','.join(map(str, bad_list))

    # Take only corrected star files if the star_corr flag is True
    if star_corr:
        corr_suffix = '_corr'

    else:
        corr_suffix = ''


    # For exact plates, do not include the scale_pattern stars
    if type_name == 'exact':
        star_regex = '_star_1*'

    else:
        star_regex = '_star*'
    


    # Run mircal
    output_file_name = type_name + '_' + site_no + type_extension
    command = 'mircal --show --site ' + site + ' --' + type_name + '=' + output_file_name + ' ' + site + \
        star_regex + corr_suffix + '.txt ' + bad_star_string

    raw_results = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE).stdout.read()

    # Return the command used to produce the results as well
    if output_command:
        return raw_results, command

    return raw_results


    
def runMirCalEliminateResiduals(plates_local_dir, mircal_parse_column_indices, site, site_no, type_name, type_extension, max_threshold, star_corr=False):

    ### OPTIMIZATION PARAMETERS - Greedy optimization
    # Minimum percent change in the residuals to continue eliminating stars
    residual_change_threshold = 0.02

    # How many stars to check after the threshold has been reaches (helps avoid local minima during optimization)
    over_threshold_counts = 2

    ###

    ### OPTIMIZATION PARAMETERS - Maximum rejection
    
    # Maximum residual threshold
    #max_threshold = 5

    # Maximum fraction of input stars to reject (0.5 = 50% of stars)
    max_rejected_stars = 0.5

    ###

    # Change the directoy to the plates dir
    os.chdir(plates_local_dir)

    bad_list = []

    # Do the initial MirCal run
    raw_results = runMirCal(site, site_no, type_name, type_extension, bad_list, star_corr=star_corr)

    printSeparator()
    print(site, type_name)
    print(raw_results.decode('unicode_escape'))
        
    # Parse mircal results into a machine readable format
    results = parseMirCalResults(raw_results, mircal_parse_column_indices)

    # Check if mircal returned any results
    if not results:
        print("ERROR! Mircal returned no results! Try manually running mircal to see what's up.")
        sys.exit()

    previous_residual = 10**5

    over_threshold_n = 0

    # all_avg_residuals = []

    # Eliminate stars which give the most improvement in residuals, until the change is too small
    while True:

        # Init the minimum residual
        min_residual = 10**5
        min_residual_ind = 0

        new_results = []

        # Go through every star and find the one which when it is eliminated gives the minimum average residual
        for i in range(len(results)):

            # Skip if star already in the bad list
            if int(results[i][0]) in bad_list:
                continue

            # Add the current star to the bad list
            bad_list.append(int(results[i][0]))

            # Run MirCal
            raw_results = runMirCal(site, site_no, type_name, type_extension, bad_list, star_corr=star_corr)
            
            # Parse mircal results into a machine readable format
            new_results = parseMirCalResults(raw_results, mircal_parse_column_indices)

            # Break if the result list is empty
            if not new_results:
                break

            # Find the average residuals
            residual_list = [line[-1] for line in new_results]
            residual_avg = sum(residual_list)/float(len(new_results))

            # Check if the current solution has the lowest residuals
            if residual_avg < min_residual:

                # Set the minimum residual as the current average
                min_residual = residual_avg

                # Find the minimum residual index
                min_residual_ind = int(results[i][0])

            # Remove the current star from the bad list
            bad_list.pop()


        print('Residual change:', min_residual, '<', (1.0 - residual_change_threshold)*previous_residual)
        
        if not new_results:
            break

        # Check if the residual change satisfes the stop conditions
        if min_residual > (1.0 - residual_change_threshold)*previous_residual:

            over_threshold_n += 1

            # Break if there were no significant changes in the residuals for N times
            if over_threshold_n >= over_threshold_counts:
                break

        else:
            over_threshold_n = 0

        # Run MirCal
        raw_results = runMirCal(site, site_no, type_name, type_extension, bad_list, star_corr=star_corr)

        # Parse mircal results into a machine readable format
        new_results = parseMirCalResults(raw_results, mircal_parse_column_indices)

        # Extract the list of residuals
        residual_list = [line[-1] for line in new_results]

        # Break if the maximum residual is below the threshold
        if max(residual_list) < max_threshold:
            break

        # TEST!!!!!!!!!!!
        # all_avg_residuals.append(min_residual)

        print('Minimum residual:', min_residual)
        print('Best index:', min_residual_ind)

        # Add the found point to the bad stars list
        bad_list.append(min_residual_ind)

        previous_residual = min_residual


    # Continue to reject residuals with maximum value until the maximum is not below a set threshold
    while len(new_results) > max_rejected_stars*len(results):

        # Run MirCal
        raw_results = runMirCal(site, site_no, type_name, type_extension, bad_list, star_corr=star_corr)
            
        # Parse mircal results into a machine readable format
        new_results = parseMirCalResults(raw_results, mircal_parse_column_indices)

        # Break if the result list is empty
        if not new_results:
            break

        # Find the maximum residual
        residual_list = [line[-1] for line in new_results]
        max_residual = max(residual_list)

        # Check if the max rejection should stop if the max value is less then the given threshold
        if max_residual < max_threshold:
            break

        bad_index = int(new_results[residual_list.index(max_residual)][0])

        print('Rejecting max:', bad_index)

        # Add the max residual to the bad list
        bad_list.append(bad_index)


    # Run mircal for the last time to get the final results
    final_result, command = runMirCal(site, site_no, type_name, type_extension, bad_list, output_command=True, star_corr=star_corr)


    print('Final results:')
    print(command)
    print(final_result.decode('unicode_escape'))

    # Store the command and the reesults to a file
    with open(site + '_' + type_name + '_command.txt', 'w') as f:
        
        # Write the command
        f.write(command)

        # Write the results
        f.write(str(final_result))



    # plt.plot(range(len(all_avg_residuals)), all_avg_residuals)
    # plt.show()

    return results


def deleteCorrectedStarFiles(plates_local_dir):
    """ Delete all star files with corrected encoder positions. """

    print('Deleting all files with corrected stars positions...')

    for file_name in os.listdir(plates_local_dir):

        if ('star' in file_name) and ('corr' in file_name):

            file_name = os.path.join(plates_local_dir, file_name)
            os.remove(file_name)
            print('Deleted: ', file_name)


def delete_badstar_hu_hv(directory):
    print(f"Deleting bad stars with abs(hu) or abs(hv) values > 250 in : {directory}")
    # Walk through all the files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a .txt file and contains 'star' in the name
            if file.endswith('.txt') and 'star' and not 'scale' in file:
                file_path = os.path.join(root, file)
                print(f"hu hv Processing file: {file_path}")
                
                # Read the content of the file
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Print the number of lines before processing
                print(f"Number of stars before hu hv processing: {len(lines)}")

                # Filter lines based on the criteria for hu and hv, but keep lines starting with '#'
                filtered_lines = []
                comment_lines = [line for line in lines if line.startswith('#')]

                for line in lines:
                    if 'hu' in line and 'hv' in line:
                        try:
                            hu_value = float(line.split('hu')[-1].split()[0])
                            hv_value = float(line.split('hv')[-1].split()[0])

                            # Keep the line only if abs(hu) <= 250 and abs(hv) <= 250
                            if abs(hu_value) <= 250 and abs(hv_value) <= 250:
                                filtered_lines.append((line, hu_value, hv_value))
                        except ValueError:
                            # In case there is an issue parsing the hu or hv value, skip the line
                            pass

                # If no lines meet the criteria, keep the best 10 lines with the smallest hu and hv
                if not filtered_lines:
                    for line in lines:
                        if 'hu' in line and 'hv' in line:
                            try:
                                hu_value = float(line.split('hu')[-1].split()[0])
                                hv_value = float(line.split('hv')[-1].split()[0])
                                filtered_lines.append((line, hu_value, hv_value))
                            except ValueError:
                                pass

                    # Sort by the sum of the absolute values of hu and hv and keep the best 10
                    filtered_lines.sort(key=lambda x: abs(x[1]) + abs(x[2]))
                    filtered_lines = filtered_lines[:10]

                # Combine the comment lines with the filtered lines
                final_lines = comment_lines + [line[0] for line in filtered_lines]

                # Write the filtered lines back to the file
                with open(file_path, 'w') as f:
                    f.writelines(final_lines)

                # Print the number of lines after processing
                print(f"Number of stars after hu hv processing: {len(filtered_lines)}")

def delete_badstar_wx_wy(directory):
    print(f"Deleting bad stars with abs(rx-wx) or abs(ry-wy) values > 2 in : {directory}")
    # Walk through all the files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a .txt file and contains 'star' in the name
            if file.endswith('.txt') and 'star' and not 'scale' in file:
                file_path = os.path.join(root, file)
                print(f"rx-wx,ry-wy Processing file: {file_path}")
                
                # Read the content of the file
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Print the number of lines before processing
                print(f"Number of stars before abs(rx-wx) and abs(ry-wy) processing: {len(lines)}")

                # Filter lines based on the criteria for abs(rx-wx) and abs(ry-wy) <= 2.0 , but keep lines starting with '#'
                filtered_lines = []
                comment_lines = [line for line in lines if line.startswith('#')]

                for line in lines:
                    if 'rx' in line and 'ry' in line and 'wy' in line and 'wy' in line:
                        try:
                            rx_value = float(line.split('rx')[-1].split()[0])
                            ry_value = float(line.split('ry')[-1].split()[0])
                            wx_value = float(line.split('wx')[-1].split()[0])
                            wy_value = float(line.split('wy')[-1].split()[0])

                            # Keep the line only if abs(rx-wx)<=2.0 and abs(ry-wy)<=2.0 
                            if abs(rx_value - wx_value) <= 2.0 and abs(ry_value - wy_value) <= 2.0:
                                #filtered_lines.append((line, abs(rx_value - wx_value), abs(ry_value -wy_value))
                                filtered_lines.append((line, rx_value, wx_value, ry_value))
                        except ValueError:
                            # In case there is an issue parsing the rx,ry,wxor wy value, skip the line
                            pass

                # If no lines meet the criteria, keep the best 10 lines with the smallest hu and hv
                if not filtered_lines:
                    for line in lines:
                        if 'rx' in line and 'ry' in line and 'wy' in line and 'wy' in line:
                            try:
                                rx_value = float(line.split('rx')[-1].split()[0])
                                ry_value = float(line.split('ry')[-1].split()[0])
                                wx_value = float(line.split('wx')[-1].split()[0])
                                wy_value = float(line.split('wy')[-1].split()[0])
                                filtered_lines.append((line, rx_value, ry_value, wx_value, wy_value))

                            except ValueError:
                                pass

                    # Sort by the sum of the absolute values of hu and hv and keep the best 10
                    #filtered_lines.sort(key=lambda x: abs(x[1]) + abs(x[2]))
                    #filtered_lines = filtered_lines[:10]

                # Combine the comment lines with the filtered lines
                final_lines = comment_lines + [line[0] for line in filtered_lines]

                # Write the filtered lines back to the file
                with open(file_path, 'w') as f:
                    f.writelines(final_lines)

                # Print the number of lines after processing
                print(f"Number of stars after rx-wx,ry-wy processing: {len(filtered_lines)}")




def generatePlates(plates_remote_dir, night_name, plates_local_dir, plates_type, sites_names, mircal_parse_column_indices, exact_no_correct=False):
    """ Automatically generate calibration plates. """

    # Copy the *star*.txt files locally
    copyStarFiles(plates_remote_dir, night_name, plates_local_dir)

    # use delete_badstar_hu_hv
    delete_badstar_hu_hv(plates_local_dir)
    # use delete_badstar_rx-wx ry-wy
    delete_badstar_wx_wy(plates_local_dir)

    # Go through each type of plate
    for type_name, type_extension, max_threshold in plates_type:

        # Run for all sites
        for site, site_no in sites_names:

            star_corr = False

            # Run the star correction before making an exact plate
            if (not exact_no_correct) and type_name == 'exact':

                # Make the scale plate name
                scale_plate_name = 'scale_'+site_no+'.aff'

                # Check if the scale plate exists
                if not os.path.isfile(os.path.join(plates_local_dir, scale_plate_name)):
                    print(scale_plate_name + ' file could not be found! Exact file will not be corrected!')

                else:

                    print('Correcting the stars file by using the scale plate:', scale_plate_name)

                    # Correct the stars file by using the scale plate
                    correctStarsFile(plates_local_dir, site, scale_plate_name)

                    # Set the flag so mircal is run only on the corrected star positions
                    star_corr = True

                    print('Star positions corrected!')


            # Delete all 'stars*corr' files if redoing the scale plate
            if type_name == 'scale':
                deleteCorrectedStarFiles(plates_local_dir)

            # Delete all 'stars*corr' files if making an exact plate with no corrections
            if (type_name == 'exact') and exact_no_correct:
                deleteCorrectedStarFiles(plates_local_dir)


            # Run MirCal
            results = runMirCalEliminateResiduals(plates_local_dir, mircal_parse_column_indices, site, site_no, type_name, type_extension, max_threshold, star_corr=star_corr)



def checkEventStatus(dir_path, event_timestamp):
    """ Checks if the event is a double station event and if it has all the neccessary files to run mirfit. 
    """

    # Format of the file timestamp
    timestamp_format = '%Y%m%d_%H%M%S'


    event_name_list = []

    # Get the list of files from in the directory
    file_list = os.listdir(dir_path)

    # Go through all the file names in the folder
    for file_name in file_list:
        
        # Check only for given extention (vid and txt)
        for extension in extensions_list:
            if extension in file_name:

                # Check if it has the right station ID
                for stat_id in station_id_list:
                    if stat_id in file_name:

                        # Add the file name to the filtered list
                        event_name_list.append(file_name)

    # Get the datetime object from the given event timestemp
    event_time = datetime.datetime.strptime(event_timestamp, timestamp_format)

    timestamp_events = []

    # Find the file with a time closest to the event time for each station
    for stat_id in station_id_list:

        # Create a list of files from only one station
        station_files = [x for x in event_name_list if stat_id in x]

        # Calcilate the time differences
        file_times_diff = [abs(event_time - datetime.datetime.strptime(file_name[3:18], timestamp_format)) \
            for file_name in station_files]

        # Continue if no files were found
        if not file_times_diff:
            continue
        
        # Find the minimum time difference index
        min_index = file_times_diff.index(min(file_times_diff))

        # Exclude the file extension
        min_file_name = station_files[min_index].split('.')[0]

        # check that min(file_times_diff) is less than 2 second
        if min(file_times_diff).total_seconds() < 2:
            # Create the file name for both txt and vid files
            for extension in extensions_list:
                timestamp_events.append(min_file_name + extension)
        else:
            # put a Warning for the event
            print(f"Event {event_timestamp} has no file for station {stat_id} within 2 seconds of the event time")

    # Calculate how many files should be found
    files_num = len(extensions_list)*len(station_id_list)
        
    paired_files = []

    # Check if the necessary files exist and add them to the return list
    for reconst_file_name in timestamp_events:
        if os.path.isfile(os.path.join(dir_path, reconst_file_name)):
            paired_files.append(reconst_file_name)

    # Check if all necessary files are here
    if len(paired_files) == files_num:
        return True, paired_files

    else:

        # Get the files which are missing from the data folder
        missing_files = list(set(timestamp_events).difference(set(paired_files)))

        return False, paired_files



def copyEventFiles(night_data_path, work_dir, event_name):
    """ Runs mirfit on the given event. """
    print('Copying files for the event:', event_name)
    print('in the night folder:', night_data_path)
    print('to the working directory:', work_dir)

    # Get the data files of the given event
    files_found, data_files = checkEventStatus(night_data_path, event_name)

    if not files_found:
        print(' ERROR! Some data files are missing from the source folder for this event!')
        print('  Night folder:', night_data_path)
        print('  Event name:', event_name)
        print('  Missing files:', data_files)
    
    work_dir_path = os.path.join(work_dir, event_name + working_dir_suffix)

    print(data_files)
    # Copy all needed files to the working dir

    # skip if files already exist ;ZK
    print(data_files[1])
    print(os.path.join(work_dir_path+'/'+data_files[1]))

    if not os.path.exists(os.path.join(work_dir_path,data_files[1])):

     for data_file in data_files:
        
        dest_path = os.path.join(work_dir_path, data_file)
        shutil.copy2(os.path.join(night_data_path, data_file), dest_path)

        # If the file is a bz2 file, unpack it
        if dest_path.endswith('.bz2'):
            print('Unpacking', dest_path)
            bz2_data = bz2.BZ2File(dest_path).read()
            with open(dest_path.strip('.bz2'), 'wb') as f:
                f.write(bz2_data)

            # Remove the bz2 file
            os.remove(dest_path)
            

    print(' ...done!')




def parseMeteorData(dir_path, file_name, header=8):
    """ Parse the CAMO meteor data. """

    unix_time = 0

    # Open the meteor data file
    with open(os.path.join(dir_path, file_name)) as f:

        # Go through the header
        for i in range(header):
            line = f.readline()

            if len(line) == 0:
                continue

            # Parse UNIX time
            if i == METEOR_HEADER_UNIX_ROW:
                line = line.replace('#', '').split()
                unix_time = float(line[1])  # Parse the UNIX time

        meteor_data = []

        # Go through the rest of the file
        while True:
            line = f.readline()
            if not line:
                break  # Break if EOF

            # Skip lines starting with '#'
            if line.startswith('#'):
                continue

            # Parse the meteor data
            try:
                meteor_data.append(list(map(float, line.replace('\n', '').split())))
            except ValueError:
                # Skip lines that cannot be converted to float
                print(f"Skipping invalid line: {line}")
                continue

        # Convert meteor data to a numpy array
        meteor_data = np.array(meteor_data)

        return unix_time, meteor_data

    # Return in case of file not found
    return 0, []


def parseStarsData(dir_path, file_name):
    """ Parse the CAMO stars data. """

    with open(os.path.join(dir_path, file_name)) as f:

        stars_data = []

        for line in f:

            # Remove the star name from the line
            line = re.sub(r"'.*?'", "", line)

            # Take every other element
            line = line.replace('\n', '').split()[1:-1:2]

            stars_data.append(map(float, line))


        # Convert stars data to a numpy array
        stars_data = np.array(stars_data)

        return stars_data

    # Return in case file not found
    return []



def plotStarsAndMeteor(dir_path, plates_local_dir, sites_names, star_corr=False):
    """ Plots the meteor's path and star positions, in encoder coordinates. """

    # Go through both stations
    for station_name, station_no in sites_names:

        station_no = int(station_no)

        # Create a subplot with an equal aspect ratio
        plt.subplot(1, 2, station_no, aspect='equal')

        # Add a title
        plt.title(station_name)


        ###########
        # Find the meteor file name
        meteor_file_name = ''

        # Go through all files in the given directory
        for file_name in os.listdir(dir_path):

            if ('ev' in file_name) and (sites_names[station_no-1][1]+'T' in file_name) and ('.txt' in file_name):
                meteor_file_name = file_name

                break

        if not meteor_file_name:
            print('No meteor file found!')
            sys.exit()


        print('Meteor file:', meteor_file_name)

        # Load meteor data
        meteor_unix_time, meteor_data = parseMeteorData(dir_path, meteor_file_name)

        # Skip plotting if meteor data is empty
        if len(meteor_data) == 0:
            return False

        # # Plot the meteor path
        # print ("meteor_data  = ", meteor_data)
        # print ("meteor_data shape  = ", len(meteor_data))
        # print ("\n\nmeteor_data[:,METEOR_Hx_INDEX] len = ", len(meteor_data[:,METEOR_Hx_INDEX]))
        # print ("meteor_data[:,METEOR_Hy_INDEX] len = ", len(meteor_data[:,METEOR_Hy_INDEX]))
        
        plt.plot(meteor_data[:,METEOR_Hx_INDEX], meteor_data[:,METEOR_Hy_INDEX], zorder=4)

        # Plot the starting point of the meteor
        plt.scatter(meteor_data[0,METEOR_Hx_INDEX], meteor_data[0,METEOR_Hy_INDEX], c='r', zorder=3)


        # Find the star files
        stars_files = []

        for file_name in os.listdir(plates_local_dir):

            if (sites_names[station_no-1][0] in file_name) and ('star' in file_name) and not (('corr' in file_name) ^ star_corr):
                stars_files.append(file_name)

        if not stars_files:
            print('No star data found!')

            sys.exit()

        print('Star files:', stars_files)

        # File name of the meteor data
        #meteor_file_name = 'ev_20161010_062505A_01T.txt'

        # File containing the stars
        # stars_files = ["Tavistock_star_1476080250.txt", "Tavistock_star_1476076492.txt", "Tavistock_star_1476084006.txt", "Tavistock_star_1476087763.txt", "Tavistock_star_1476091518.txt"]

        stars_data = []

        # Go through all star files and parse the data
        for stars_file in stars_files:
            
            header, star = loadStarsData(plates_local_dir, stars_file)

            if (not star) or (star is None):
                continue

            # Load stars data
            stars_data.append(star)


        # Find the stars file that was taken closest to the meteor event
        star_times = np.array([(stars[0][0], stars[-1][0]) for stars in stars_data])
        star_times -= meteor_unix_time
        star_times = np.abs(star_times)

        closest_star_file_index = int(np.argmin(star_times)/2)

        print('Closest star file:', stars_files[closest_star_file_index])

        # Go through the star data and plot them
        for i, stars in enumerate(stars_data):

            # Convert stars to a float numpy array and don't include the name
            stars = np.array(stars)[:, :-2].astype(np.float64)

            # Set a special color for the stars that are closest to the meteor event
            if i == closest_star_file_index:
                star_color = 'b'

            else:
                star_color = '0.5'

            # Plot the stars
            plt.scatter(stars[:, STAR_Hx_INDEX] + stars[:, STAR_Hu_INDEX], stars[:, STAR_Hy_INDEX] + stars[:, STAR_Hv_INDEX], facecolors='none', edgecolors=star_color, s=(i*5 + 20))


        # Set background color to black
        plt.gca().set_facecolor('black')


        # Set plot limits
        plt.xlim((0, 2**16))
        plt.ylim((0, 2**16))

        plt.gca().tick_params(axis='x', labelsize=7)
        plt.gca().tick_params(axis='y', labelsize=7)

        plt.gca().set_axisbelow(True)
        plt.grid(color='0.5')


    # Add a suptitle with the name of the event
    plt.suptitle(os.path.split(dir_path)[-1])
    plt.subplots_adjust(top=1.2)

    #plt.tight_layout()

    # Save the image to the event directory
    plt.savefig(os.path.join(dir_path, 'star_coverage.png'), bbox_inches='tight', dpi=300)

    # plt.show()
    plt.close()

def CorectAstAffName(file_name_ext,num,ext,work_dir, plates_workdir_prefix, night_name):
    if not os.path.exists(os.path.join(work_dir, plates_workdir_prefix+night_name, file_name_ext)):
        # find the files that have the .ast extension and 01 in the name
        for file_name in os.listdir(os.path.join(work_dir, plates_workdir_prefix+night_name)):
            if (num in file_name) and (ext in file_name):
                # rename the file to the exact_01.ast
                os.rename(os.path.join(work_dir, plates_workdir_prefix+night_name, file_name), os.path.join(work_dir, plates_workdir_prefix+night_name, file_name_ext))
                print(file_name,'file found and renamed',file_name_ext)
                break


if __name__ == "__main__":

    print('Starting MirfitPrepare...')


    # List of events to be processed
    event_list = []

    # Handle command line arguments
    arg_parser = argparse.ArgumentParser(description="""An automated script for handling and calibrating CAMO 
        data. Author: Denis Vida, e-mail: dvida@uwo.ca, (c) 2016
        Edited 2024-05-14: Maximilian Vovk, email: mvovk@uwo.ca = check the plates_reductions folder before computing a new plate
        Edited 2024-05-24: Maximilian Vovk, email: mvovk@uwo.ca = check the automaticaly created plates in plates_reductions_auto folder before computing a new plate and renames the .ast and .aff files if maual plates have different names
        """)
    arg_parser.add_argument('meteors', help='Comma separated meteor events.')
    # arg_parser.add_argument('meteors', default='20240715_050815', help='Comma separated meteor events.')

    # Add a mutually exclusive for the parser
    argparse_cal_group = arg_parser.add_mutually_exclusive_group()
    argparse_cal_group.add_argument('-c', '--calibration', metavar='CALIBRATION_PATH', help="""Use existing calibration, give the path to the
        directory containing calibration files.""")
    argparse_cal_group.add_argument('-d', '--default-plates', default=False, const=True, action='store_const', help='Use the default plates given in the plates_default directory.')

    arg_parser.add_argument('-e', '--exact', default=False, const=True, action='store_const', help='Make a new exact plate from the original star files.')
    arg_parser.add_argument('-s', '--scale', default=False, const=True, action='store_const', help='Make a new scale plate from the original star files. Note: the exact plate will be redone as well if the -x option is not turned on!')
    arg_parser.add_argument('-n', '--no-correct', default=False, const=True, action='store_const', help="Do not perform 'virtual' mirror correction on the star files, using the scale plate.")
    arg_parser.add_argument('-x', '--hide-plot', default=False, const=True, action='store_const', help="Do not show the plot of star positions and the meteor's path.")
    
    # add the option to change the MAX_ERR_STAR
    arg_parser.add_argument('--max_err_star', type=int, default=3, \
        help="""The maximum error star. Default is 3.""")
    # add the option to change the MIN_STAR_COUNT
    arg_parser.add_argument('--min_star_count', type=int, default=15, \
        help="""The minimum number of stars. Default is 15.""")
    # add the option to change the PERCENTILE_BAD_STAR
    arg_parser.add_argument('--percentile_bad_star', type=int, default=99, \
        help="""The percentile of the bad stars. Default is 99.""")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Override the event list if there are events given in the command line arguments
    if cml_args.meteors:
        event_list = cml_args.meteors.split(',')

    # Check if the calibration directory is provided
    calibration_dir_provided = ''
    if cml_args.calibration:
        
        calibration_dir_provided = os.path.abspath(cml_args.calibration)

        print('Using the provided plates directory:', calibration_dir_provided)



    # Format event names to a uniform format
    event_list = formatEventNames(event_list)

    print('Processing events:', event_list)

    # Create the local working directory if it does not exist
    mkdirP(work_dir)

    # Extract night directories from the event list
    night_list = extractNights(event_list)

    load_plate_auto=False
    load_plate=False
    # Go through each night
    for night_name in night_list:

        printSeparator()

        print('Running night', night_name)

        # Take only those events in the current night
        night_event_list = [event for event in event_list if event.startswith(night_name)]

        # Check if there are any events to process
        if not night_event_list:
            
            print('No events to process! Exiting...')
            break


        # Check if the calibration directory was specified in the commandline args
        if calibration_dir_provided:

            print('Using provided calibration files...')

            plates_local_dir = calibration_dir_provided

        else:
            
            print('Using the default calibration directory...')

            if os.path.exists(os.path.join(work_dir, plates_workdir_prefix+night_name)):
                print('Plate folder already exists in the working directory...')


            
    
        # Check if the default calibration is present if requested
        if cml_args.default_plates:

            print('Using calibration from the default dir...')

            # Path of the default calibration dir
            plates_default_path = os.path.join(work_dir, plates_default_dir)

            if not os.path.exists(plates_default_dir):
                
                print('The default plates directory does not exits:', plates_default_path)
                sys.exit()

            # Define the directory which contains plates
            plates_local_dir = os.path.join(work_dir, plates_workdir_prefix+night_name)

            # Make the plate directory if it does not exits
            mkdirP(plates_local_dir)

            scale_list = []
            exact_list = []

            # Search for the plate files
            for file_name in os.listdir(plates_default_path):

                if ('scale' in file_name) and ('01' in file_name):
                    scale_list.append(file_name)

                elif ('scale' in file_name) and ('02' in file_name):
                    scale_list.append(file_name)

                elif ('exact' in file_name) and ('01' in file_name):
                    exact_list.append(file_name)

                elif ('exact' in file_name) and ('02' in file_name):
                    exact_list.append(file_name)


            find_scale = False

            # Check if there are enough scale plate files
            if len(scale_list) == len(plates_type):

                # Copy the scale plates to the plates dir
                for file_name in scale_list:
                    shutil.copy2(os.path.join(plates_default_path, file_name), os.path.join(plates_local_dir, file_name))
                
            else:

                print('Scale plates missing from the default directory, trying to find them in the night dir...')
                
                find_scale = True
                cml_args.default_plates = False


            # Check if there are enough exact plate files
            if len(exact_list) == len(plates_type):

                # Copy the exact plates to the plates dir
                for file_name in exact_list:
                    shutil.copy2(os.path.join(plates_default_path, file_name), os.path.join(plates_local_dir, file_name))

            else:

                if not find_scale:

                    # Redo the exact plates if the scale plates were found
                    cml_args.exact = True

                else:
                    print('Exact plates missing from the default directory, trying to find them in the night dir...')

                cml_args.default_plates = False



        # Run in case of normal operations
        if not (cml_args.calibration or cml_args.default_plates):
            # Run when no calibration directory was provided

            # Define the directory which contains plates
            plates_local_dir = os.path.join(work_dir, plates_workdir_prefix+night_name)

            # check if is present in plates_reductions a folder with the name of the night and if so copy the folder into the working directory
            if os.path.exists(os.path.join(plates_reductions, plates_workdir_prefix+night_name)):
                print('Copying the calibration files from the',plates_reductions,'directory...')
                shutil.copytree(os.path.join(plates_reductions, plates_workdir_prefix+night_name), os.path.join(work_dir, plates_workdir_prefix+night_name))
                # check that a file exact_01.ast might have a different name in the folder
                CorectAstAffName('exact_01.ast','01','.ast',work_dir, plates_workdir_prefix, night_name)
                CorectAstAffName('exact_02.ast','02','.ast',work_dir, plates_workdir_prefix, night_name)
                CorectAstAffName('scale_01.aff','01','.aff',work_dir, plates_workdir_prefix, night_name)
                CorectAstAffName('scale_02.aff','02','.aff',work_dir, plates_workdir_prefix, night_name)
                    
                print('copy done!')
                load_plate=True
            
            if load_plate==False:
                # prioritize the data in the manual reduction folder
                if os.path.exists(os.path.join(plates_reductions_auto, plates_workdir_prefix+night_name)):
                    print('Copying the calibration files from the',plates_reductions_auto,'directory...')
                    shutil.copytree(os.path.join(plates_reductions_auto, plates_workdir_prefix+night_name), os.path.join(work_dir, plates_workdir_prefix+night_name), dirs_exist_ok=True) # Allow copying into an existing directory
                    print('WARNING: The plate has been copied from the automatic plate folder!')
                    load_plate_auto=True
                    load_plate=True

            if load_plate==False:
                # Check if the calibration was overridden, and make new calibration files
                if cml_args.exact or cml_args.scale:

                    override_plates_list = []

                    plates_type_special = []

                    # If redoing the scale plate, the exact plate must be redone as well
                    if cml_args.scale:

                        plates_type_special.append(plates_type[0])
                        override_plates_list.append('scale')

                        # Skip redoing the exact plate only if no corrections will be applied to the star files
                        if not cml_args.no_correct:
                            plates_type_special.append(plates_type[1])
                            override_plates_list.append('exact')

                    if cml_args.exact:
                        plates_type_special.append(plates_type[1])
                        override_plates_list.append('exact')


                    print('Calibration overridden, making new calibration files:', ', '.join(override_plates_list))


                    # Generate the plates
                    generatePlates(plates_remote_dir, night_name, plates_local_dir, plates_type_special, 
                        sites_names, mircal_parse_column_indices, cml_args.no_correct)


                else:
                    # Use the existing calibration if available, if not, make a new one

                    # Check for existing calibration
                    plate_files = findCalibrationFiles(plates_local_dir, plates_type, sites_names)

                    # Use the existing calibration
                    if len(plate_files) == 4:

                        print('Using the existing calibration...')


                    else:
                        # Make a new calibration

                        # Generate the plates
                        generatePlates(plates_remote_dir, night_name, plates_local_dir, plates_type, sites_names,
                            mircal_parse_column_indices)


        # Get plate file names
        plate_files = findCalibrationFiles(plates_local_dir, plates_type, sites_names)

        if len(plate_files) != 4:
            print('Error during plate generation - some plate files are missing:', plate_files)

            sys.exit()



        # Go through all given events in this night
        for event_name in night_event_list:

            print('Processing event', event_name)

            # Create the night working directory
            mkdirP(os.path.join(work_dir, event_name+working_dir_suffix))

            # Copy all output plate files to the working directory
            work_dir_path = os.path.join(work_dir, event_name+working_dir_suffix)

            for output_plate_file in plate_files:
                shutil.copy2(os.path.join(plates_local_dir, output_plate_file), 
                    os.path.join(work_dir_path, output_plate_file))


            # Copy files of the currenty processed event
            copyEventFiles(os.path.join(data_path, night_name), work_dir, event_name)



            print('Looking for',os.path.join(work_dir, 'MirfitPrepare_recalibrate_plates.py'))

            if os.path.exists(os.path.join(work_dir, 'MirfitPrepare_recalibrate_plates.py')) and load_plate==True : # load_plate=False
                # check if in the same folder is also present MirfitPrepare_recalibrate_plates.py
                print('Recalibrate the plates, running MirfitPrepare_recalibrate_plates...')
                # Run the processEvents.py script
                processEvents(event_list,cml_args.max_err_star,cml_args.min_star_count,cml_args.percentile_bad_star)
            else:
                print('MirfitPrepare_recalibrate_plates.py not found, will not recalibrate the plates...')

            if not cml_args.hide_plot:
                
                # Show plots of meteors' path and stars
                plotStarsAndMeteor(work_dir_path, plates_local_dir, sites_names, not cml_args.no_correct)
        
        if load_plate_auto==True:
            printSeparator()
            print('WARNING: The plate has been copied from the automatic plate folder!')


        

