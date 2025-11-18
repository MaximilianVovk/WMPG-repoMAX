""" Collecting and organizing EMCCD dump files and calibration plates. """

from __future__ import print_function, absolute_import, division

import os
import errno
import glob
import shutil
import argparse
import subprocess
import tarfile
import bz2
import datetime
import platform
import re

# Version number
__version__ = 1.0

# Local working directory
WORK_DIR = os.getcwd()


# Data path variables EMCCD
DATA_PATH_EMCCD  = '/srv'+os.sep+'meteor'+os.sep+'emccd'+os.sep+'events'+os.sep
FLATS_PATH_EMCCD = '/srv'+os.sep+'meteor'+os.sep+'emccd'+os.sep+'calib'+os.sep+'auto-flats'+os.sep
BIAS_PATH_EMCCD  = '/srv'+os.sep+'meteor'+os.sep+'emccd'+os.sep+'calib'+os.sep+'auto-bias'+os.sep
PLATES_PATH_EMCCD = '/srv'+os.sep+'meteor'+os.sep+'emccd'+os.sep+'plcheck'+os.sep
SKYFIT2_PATH_EMCCD = '/srv'+os.sep+'meteor'+os.sep+'reductions'+os.sep+'config+calib'+os.sep+'skyfit2'+os.sep+'emccd'+os.sep
# SKYFIT2_PATH_EMCCD = '/srv'+os.sep+'meteor'+os.sep+'emccd'+os.sep+'skyfit2'+os.sep
SITES_EMCCD = [['tavis_F', '1', 'F'], ['elgin_F', '2', 'F'],
        ['tavis_G', '1', 'G'], ['elgin_G', '2', 'G']]

# Data path variables KLINGON
DATA_PATH_KLINGON = '/srv'+os.sep+'meteor'+os.sep+'klingon'+os.sep+'events'+os.sep
FLATS_PATH_KLINGON = '/srv'+os.sep+'meteor'+os.sep+'klingon'+os.sep+'flats'+os.sep
PLATES_PATH_KLINGON = '/srv'+os.sep+'meteor'+os.sep+'klingon'+os.sep+'plcheck'+os.sep 
SKYFIT2_PATH_KLINGON = '/srv'+os.sep+'meteor'+os.sep+'reductions'+os.sep+'config+calib'+os.sep+'skyfit2'+os.sep+'klingon'+os.sep
SITES_KLINGON = [['tavis_K', 1], ['elgin_K', 2]]


# PLATES_SUFFIX_EMCCD = '_plate.ast'
PLATE_FIRST= 'first.ast' # fit_20090715_01K.ast
FLAT_FILE_NAME = 'flat.png' # flat_20221027_01K.png
BIAS_FILE_NAME = 'bias.png'

WORK_DIR_SUFFIX = '_skyfit2'

# check the OS
if platform.system() == 'Windows':
    print('Windows OS detected')
    # change the /srv with M:
    DATA_PATH_EMCCD = 'M:'+os.sep+'emccd'+os.sep+'events'+os.sep
    FLATS_PATH_EMCCD = 'M:'+os.sep+'emccd'+os.sep+'calib'+os.sep+'auto-flats'+os.sep
    BIAS_PATH_EMCCD = 'M:'+os.sep+'emccd'+os.sep+'calib'+os.sep+'auto-bias'+os.sep
    PLATES_PATH_EMCCD = 'M:'+os.sep+'emccd'+os.sep+'plcheck'+os.sep
    SKYFIT2_PATH_EMCCD = 'M:'+os.sep+'reductions'+os.sep+'config+calib'+os.sep+'skyfit2'+os.sep+'emccd'+os.sep
    # SKYFIT2_PATH_EMCCD = 'M:'+os.sep+'emccd'+os.sep+'skyfit2'+os.sep
    DATA_PATH_KLINGON = 'M:'+os.sep+'klingon'+os.sep+'events'+os.sep
    FLATS_PATH_KLINGON = 'M:'+os.sep+'klingon'+os.sep+'flats'+os.sep
    PLATES_PATH_KLINGON = 'M:'+os.sep+'klingon'+os.sep+'plcheck'+os.sep
    SKYFIT2_PATH_KLINGON = 'M:'+os.sep+'reductions'+os.sep+'config+calib'+os.sep+'skyfit2'+os.sep+'klingon'+os.sep
elif platform.system() == 'Linux':
    print('Linux OS detected')
else:
    print('OS not recognized, please check the paths in the code')



class ErrorTracker(object):
    """ Indicates if any errors or warnings occured during the running of the program. """

    def __init__(self):

        self.errors = []


    def addError(self, error_msg):

        self.errors.append(error_msg)
        print(error_msg)

def adjust_event_time(event_name, seconds_add_sub=0):
    """ Adjusts the event time by adding or subtracting a specified number of seconds.

    Args:
    event_name (str): The event time as a string in the format 'YYYYMMDD_HHMMSS'.
    seconds (int): The number of seconds to adjust. Positive adds seconds, negative subtracts.

    Returns:
    str: The new event time in the same format.
    """
    # Parse the event_name to a datetime object
    event_datetime = datetime.datetime.strptime(event_name, '%Y%m%d_%H%M%S')
    
    # Adjust by the specified number of seconds
    event_datetime += datetime.timedelta(seconds=seconds_add_sub)
    
    # Convert the datetime object back to a string
    new_event_name = event_datetime.strftime('%Y%m%d_%H%M%S')
    return new_event_name

def mkdirP(path):
    """ Make a directory if it does not exist. """

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



def regexCopy(source, destination):
    """ Copy all the files which match the source regex to destination. """

    # Get a list of files satisfying the regex
    file_list = glob.glob(source)

    for file_path in file_list:
        
        # Get only the name of the file, not the whole path
        _, file_name = os.path.split(file_path)

        # Copy the file to destination
        shutil.copy2(file_path, os.path.join(destination, file_name))


    # Return the copied files list
    return file_list



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



def tarExtract(source, tar_file, site_dir):
    """Extracts files from tar'd directory."""
    
    tar_file_name = os.path.join(source, tar_file)
    tar_dir_name = tar_file_name.rstrip('.tar')

    # Unpack the dump files from tar
    print('Unpacking', tar_file_name)
    with tarfile.open(tar_file_name) as tar_data:
        tar_data.extractall(path=source)
    
    # Move extracted files to site_dir, overwriting existing files
    for file in os.listdir(tar_dir_name):
        file_path = os.path.join(tar_dir_name, file)
        dest_path = os.path.join(site_dir, file)
        if os.path.exists(dest_path):
            os.remove(dest_path)  # Overwrite existing file only
        shutil.move(file_path, site_dir)
        
    os.rmdir(tar_dir_name)


def runVidchop(source, destination_dir):
    """ Runs vidchop on the given source file, and stores the results in the destination folder. """

    # Add a dir separator at the end, if it is not already there
    if destination_dir[-1] != os.sep:
        destination_dir += os.sep

    # Generate vidchop command which will be run in the terminal
    command = "vidchop --flush \"" + destination_dir + "\" \"" + source + "\""

    print(command)


    # Run vidchop command
    raw_results = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE).stdout.read()

    print(raw_results)



def copyFlats(night_name, site_id, destination_dir, flats_path, file_name_search=''):
    """ Checks for existence of flats and copies them to appropriate site directory. """

    flats_night_path = os.path.join(flats_path, night_name)

    #check if the flats exist considering th

    # Generate the flat image regex
    flat_regex = os.path.join(flats_night_path, 'flat*' + night_name + '*' + str(site_id) + '.png')

    # Check if such a file exists
    flats_found = glob.glob(flat_regex)

    if not flats_found and file_name_search!='':
        # check if file_name_search is in the folder
        flat_regex = os.path.join(flats_night_path, file_name_search)
        flats_found = glob.glob(flat_regex)

    # If the flats were found, copy them to the local directory
    if flats_found:

        flat_name = flats_found[0]

        # Copy the flat file to the site directory
        shutil.copy2(os.path.join(flats_night_path, flat_name), os.path.join(destination_dir, FLAT_FILE_NAME))

        return True

    else:
        return False
    
    
    
def copyBias(night_name, site_id, destination_dir, bias_path, file_name_search=''):
    """ Checks for existence of flats and copies them to appropriate site directory. """

    bias_night_path = os.path.join(bias_path, night_name)

    # Generate the flat image regex
    bias_regex = os.path.join(bias_night_path, 'bias*' + night_name + '*' + str(site_id) + '.png')

    # Check if such a file exists
    bias_found = glob.glob(bias_regex)

    if not bias_found and file_name_search!='':
        # check if file_name_search is in the folder
        bias_regex = os.path.join(bias_night_path, file_name_search)
        bias_found = glob.glob(bias_regex)

    # If the flats were found, copy them to the local directory
    if bias_found:

        bias_name = bias_found[0]

        # Copy the flat file to the site directory
        shutil.copy2(os.path.join(bias_night_path, bias_name), os.path.join(destination_dir, BIAS_FILE_NAME))

        return True

    else:
        return False
    


def copyPlates(night_name, site_id, destination_dir,plates_path, file_name_search=''):
    """ Checks for existence of plates and copies them to appropriate site directory. """

    plates_night_path = os.path.join(plates_path, night_name)

    # Generate the plate image regex
    plate_regex = os.path.join(plates_night_path, 'fit*' + night_name + '*' + str(site_id) + '.ast')

    # Check if such a file exists
    plates_found = glob.glob(plate_regex)

    if not plates_found and file_name_search!='':
        # check if file_name_search is in the folder
        plate_regex = os.path.join(plates_night_path, file_name_search)
        plates_found = glob.glob(plate_regex)

    # If the plates were found, copy them to the local directory
    if plates_found:

        plate_name = plates_found[0]

        # Copy the plate file to the site directory
        shutil.copy2(os.path.join(plates_night_path, plate_name), os.path.join(destination_dir, PLATE_FIRST))

        return True

    else:
        return False



def processEMCCDEvents(event_list):
    """ Creates the folder structure for METAL, and fetches the data and calibration files. """

    for event_name in event_list:
        
        # Create a directory for the event
        event_dir =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)

        # Make the event directory
        mkdirP(event_dir)


        # Get the name of the night of the event
        night_name = extractNights([event_name])[0]

        # save the initial event name in case the files are not found in that second
        event_name_intitial=event_name
        change_event_name = False

        for second_adjust in [0,1,-1]:

            # Regex for all CAMO widefiled files for this event
            files_regex = 'ev_' + adjust_event_time(event_name, second_adjust) + '*.tar'
            file_regex_dir = os.path.join(DATA_PATH_EMCCD, night_name, files_regex)
            if glob.glob(file_regex_dir):
                if second_adjust != 0:
                    event_name = adjust_event_time(event_name, second_adjust)
                    change_event_name = True
                print('Copying files :', file_regex_dir)

                # Copy files of the event from all SITES_EMCCD
                copied_files = regexCopy(file_regex_dir, os.path.join(event_dir))

                print('Copied files:')
                for cp_file_name in copied_files:
                    print('  ' + cp_file_name)



        # Go through all SITES_EMCCD
        for site_name, site_num, cam_code in SITES_EMCCD:

            # Make a site directory
            site_dir = os.path.join(event_dir, site_name)
            mkdirP(site_dir)

            event_site_file_found = False
            # Find the appropriate raw file to use for vidchop
            for event_site_file_name in os.listdir(event_dir):

                # Check if this is the file from the given station
                if (event_name in event_site_file_name) and (str(site_num) \
                    + cam_code + '.tar' in event_site_file_name):
                    if change_event_name:
                        error_track.addError(event_name_intitial +': WARNING! Data file for site: ' + site_name + ' not found!')
                        error_track.addError(event_name_intitial+': SOLVED! Found data for ' + site_name+' for a different second '+event_name_intitial+'->'+event_name)

                    event_site_file_found = True
                    break

                if event_name!=event_name_intitial:
                    if (event_name_intitial in event_site_file_name) and (str(site_num) \
                    + cam_code + '.tar' in event_site_file_name):

                        event_site_file_found = True
                        break



            # Check if the file was found
            if not event_site_file_found:
                error_track.addError(event_name_intitial +': WARNING! Data file for site: ' + site_name + ' not found!')
                os.rmdir(site_dir)
                continue

            else:
                print('Data file for site: ' + site_name + ' successfully copied!')



            # Extract dump and plate files from tar
            tarExtract(event_dir, event_site_file_name, site_dir)

            # Copy flat image
            copy_flat_status = copyFlats(night_name, site_num+cam_code, site_dir,FLATS_PATH_EMCCD)

            if not copy_flat_status:
                error_track.addError(event_name_intitial +': WARNING! Flat file for site: ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_flat_status = copyFlats(previous_night_name, site_num+cam_code, site_dir,FLATS_PATH_EMCCD)
                    if copy_flat_status:
                        error_track.addError(event_name_intitial +': SOLVED! Found valid Flat for ' + site_name + ' in : ' +FLATS_PATH_EMCCD+''+ previous_night_name)
                        break
                    if previous_day==29:
                        error_track.addError(event_name_intitial +': No Flat found for ' + site_name + ' in the last 30 days, try look in : '+FLATS_PATH_EMCCD+''+ night_name )

            else:
                print('Flat file for site: ' + site_name + ' successfully copied!')


            # Copy bias image
            copy_bias_status = copyBias(night_name, site_num+cam_code, site_dir,BIAS_PATH_EMCCD)

            if not copy_bias_status:
                error_track.addError(event_name_intitial +': WARNING! Bias file for site: ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_bias_status = copyBias(previous_night_name, site_num+cam_code, site_dir,BIAS_PATH_EMCCD)
                    if copy_bias_status:
                        error_track.addError(event_name_intitial +': SOLVED! Found valid Bias for ' + site_name + ' in : '+BIAS_PATH_EMCCD+''+ previous_night_name)
                        break
                    if previous_day==29:
                        error_track.addError(event_name_intitial +': No Bias found for ' + site_name + ' in the last 30 days, try look in : '+BIAS_PATH_EMCCD+''+ night_name )

            else:
                print('Bias file for site: ' + site_name + ' successfully copied!')
                

            # Copy the plates
            copy_plates_status = copyPlates(night_name, site_num+cam_code, site_dir, PLATES_PATH_EMCCD)
            
            if not copy_plates_status:
                error_track.addError(event_name_intitial +': WARNING! Plate for site ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_plates_status = copyPlates(previous_night_name, site_num+cam_code, site_dir,PLATES_PATH_EMCCD)
                    if copy_plates_status:
                        error_track.addError(event_name_intitial +': SOLVED! Found valid Plate for ' + site_name + ' in : '+PLATES_PATH_EMCCD+''+ previous_night_name)
                        break
                    if previous_day==29:
                        error_track.addError(event_name_intitial +': No Plate found for ' + site_name + ' in the last 30 days, try look in : '+PLATES_PATH_EMCCD+''+ night_name )

            else:
                print('Plate for site: ' + site_name + ' successfully copied!')
        
  

def processKlingonEvents(event_list):
    """ Creates the folder structure for METAL, and fetches the data and calibration files. """

    for event_name in event_list:
        
        # Create a directory for the event
        event_dir =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)

        # Make the event directory
        mkdirP(event_dir)


        # Get the name of the night of the event
        night_name = extractNights([event_name])[0]

        # save the initial event name in case the files are not found in that second
        event_name_intitial=event_name
        change_event_name = False

        # try the first time to get the files if doesnt work try the previous second or early second (adjust_event_time(event_name, 1))
        for second_adjust in [0,1,-1]:
            # Regex for all CAMO widefiled files for this event
            files_regex = 'ev_' + adjust_event_time(event_name, second_adjust) + '*K*.vid.bz2'
            file_regex_dir = os.path.join(DATA_PATH_KLINGON, night_name, files_regex)

            # check if the files are found
            if glob.glob(file_regex_dir):
                if second_adjust != 0:
                    event_name = adjust_event_time(event_name, second_adjust)
                    change_event_name = True

                print('Copying files :', file_regex_dir)

                # Copy files of the event from all SITES_KLINGON
                copied_files = regexCopy(file_regex_dir, os.path.join(event_dir))

                print('Copied files:')
                for cp_file_name in copied_files:
                    print('  ' + cp_file_name)

                break



        # Go through all SITES_KLINGON
        for site_name, site_id in SITES_KLINGON:

            # Make a site directory
            site_dir = os.path.join(event_dir, site_name)
            mkdirP(site_dir)

            event_site_file_found = False

            # Find the appropriate raw file to use for vidchop
            for event_site_file_name in os.listdir(event_dir):

                # Check if this is the file from the given station
                if (event_name in event_site_file_name) and (str(site_id) \
                    + 'K.vid.bz2' in event_site_file_name):
                    if change_event_name:
                        error_track.addError(event_name_intitial +': WARNING! Data file for site: ' + site_name + ' not found!')
                        error_track.addError(event_name_intitial+': SOLVED! Found data for ' + site_name+' for a different second '+event_name_intitial+'->'+event_name)

                    event_site_file_found = True
                    break

                if event_name!=event_name_intitial:
                    if (event_name_intitial in event_site_file_name) and (str(site_id) \
                    + 'K.vid.bz2' in event_site_file_name):

                        event_site_file_found = True
                        break


            # Check if the file was found
            if not event_site_file_found:
                error_track.addError(event_name_intitial +': WARNING! Data file for site: ' + site_name + ' not found!')
                os.rmdir(site_dir)
                continue

            else:
                print('Data file for site: ' + site_name + ' successfully copied!')


            bz2_file_name = os.path.join(event_dir, event_site_file_name)
            vid_file_name = bz2_file_name.strip('.bz2')

            # Unpack the vid file from bz2
            print('Unpacking', bz2_file_name)
            bz2_data = bz2.BZ2File(bz2_file_name).read()
            with open(vid_file_name, 'wb') as f:
                f.write(bz2_data)

            # Run vidchop on the event vid file
            runVidchop(vid_file_name, site_dir)

            # Delete the vid file
            os.remove(vid_file_name)


            # Copy flat image
            copy_flat_status = copyFlats(night_name, site_id, site_dir,FLATS_PATH_KLINGON, 'flat_'+night_name+'_0'+str(site_id)+'K.png')
            print('flat_'+night_name+'_0'+str(site_id)+'K.png')

            if not copy_flat_status:
                error_track.addError(event_name_intitial +': WARNING! Flat file for site: ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_flat_status = copyFlats(previous_night_name, site_id, site_dir,FLATS_PATH_KLINGON, 'flat_'+previous_night_name+'_0'+str(site_id)+'K.png')
                    if copy_flat_status:
                        error_track.addError(event_name_intitial +': SOLVED! Found valid Flat for ' + site_name + ' in : ' +FLATS_PATH_KLINGON+''+ previous_night_name )
                        break
                    if previous_day==29:
                        error_track.addError(event_name_intitial +': No Flat found for ' + site_name + ' in the last 30 days, try look in : '+FLATS_PATH_KLINGON+''+ night_name )

            else:
                print('Flat file for site: ' + site_name + ' successfully copied!')


            # Copy the plates
            copy_plates_status = copyPlates(night_name, site_id, site_dir, PLATES_PATH_KLINGON, 'fit_'+night_name+'_0'+str(site_id)+'K.ast')  # fit_20090715_01K.ast
            print('fit_'+night_name+'_0'+str(site_id)+'K.ast')
            
            if not copy_plates_status:
                error_track.addError(event_name_intitial +': WARNING! Plate for site ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_plates_status = copyPlates(previous_night_name, site_id, site_dir,PLATES_PATH_KLINGON, 'fit_'+previous_night_name+'_0'+str(site_id)+'K.ast')
                    if copy_plates_status:
                        error_track.addError(event_name_intitial +': SOLVED! Found valid Plate for ' + site_name + ' in : ' +PLATES_PATH_KLINGON+''+ previous_night_name )
                        break
                    if previous_day==29:
                        error_track.addError(event_name_intitial +': No Plate found for ' + site_name + ' in the last 30 days, try look in : '+PLATES_PATH_KLINGON+''+ night_name )

            else:
                print('Plate for site: ' + site_name + ' successfully copied!')


def get_event_datetime(event_name):
    """Extract datetime from event name like '20190506_033100'."""
    # First try the simple case: YYYYMMDD_HHMMSS at the start
    try:
        return datetime.datetime.strptime(event_name[:15], "%Y%m%d_%H%M%S")
    except Exception:
        # Fallback: search anywhere in the string
        m = re.search(r"(\d{8}_\d{6})", event_name)
        if m:
            return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        return None


def find_best_cal_file(station, event_dt):
    """
    Given a station code like '02G' and the event datetime, return the path
    to the best .cal file in SKYFIT2_PATH_EMCCD/station.

    Logic:
    - Use the latest dated file (XX_platepar_YYYYMMDD_*.cal) with date <= event date.
    - If event is earlier than all dated .cal files and XX_platepar.cal exists,
      use XX_platepar.cal (unlabeled = "oldest").
    - If there are no dated files but XX_platepar.cal exists, use that.
    - If event_dt is None: prefer XX_platepar.cal, else latest dated.
    """

    station_dir = os.path.join(SKYFIT2_PATH_EMCCD, station)

    # Unlabeled "oldest" cal, e.g. 01F_platepar.cal
    default_name = f"{station}_platepar.cal"
    default_path = os.path.join(station_dir, default_name)
    has_default = os.path.exists(default_path)

    candidates = []

    # Look for files like 02G_platepar_YYYYMMDD_*.cal
    try:
        files = os.listdir(station_dir)
    except FileNotFoundError:
        return default_path if has_default else None

    for fname in files:
        if not fname.startswith(f"{station}_platepar_") or not fname.endswith(".cal"):
            continue

        # Grab first YYYYMMDD in the filename (works with *_512x512.cal, *_newFOV.cal, etc.)
        m = re.search(r"(\d{8})", fname)
        if not m:
            continue

        date_str = m.group(1)
        try:
            cal_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            continue

        candidates.append((cal_date, fname))

    # If we don't have an event datetime, just pick a reasonable default
    if event_dt is None:
        if has_default:
            return default_path
        if candidates:
            # latest dated cal
            _, best_fname = max(candidates, key=lambda x: x[0])
            return os.path.join(station_dir, best_fname)
        return None

    event_date = event_dt.date()

    if candidates:
        # Dated cals on or before event date
        valid = [c for c in candidates if c[0] <= event_date]

        if valid:
            # Latest one before/at event
            _, best_fname = max(valid, key=lambda x: x[0])
            return os.path.join(station_dir, best_fname)
        else:
            # Event earlier than all dated cals:
            #   - if unlabeled exists, treat it as oldest
            #   - otherwise, use the earliest dated cal
            if has_default:
                return default_path
            _, best_fname = min(candidates, key=lambda x: x[0])
            return os.path.join(station_dir, best_fname)

    # No dated .cal files, but we might have the unlabeled "oldest"
    if has_default:
        return default_path

    return None









if __name__ == "__main__":

    # Tracks errors during the running of the program
    error_track = ErrorTracker()


    # List of events to be processed
    event_list = []

    # Handle command line arguments
    arg_parser = argparse.ArgumentParser(description="""An automated script for
        collecting EMCCD dump and calibration files. 
        Author: Denis Vida, e-mail: dvida@uwo.ca, (c) 2017. 
        Edited 2023: Peter Quigley, email: pquigle@uwo.ca.
        Edited 2023-04-25: Maximilian Vovk, email: mvovk@uwo.ca = now the code look for previos days bias flat and plate and rename plate.ast as first.ast if there is no first.ast file and delete .tar files
        Edited 2023-09-19: Maximilian Vovk, email: mvovk@uwo.ca = add skyfit2 .config and .cal to the event directory
        Edited 2024-05-13: Maximilian Vovk, email: mvovk@uwo.ca = add get also the Klingon files with config and cal for skyfit2 and check previous seconds or early seconds if the file is not found
        Edited 2024-06-06: Maximilian Vovk, email: mvovk@uwo.ca = no longer break if there are data in the F or G folder
        Edited 2025-11-17: Maximilian Vovk, email: mvovk@uwo.ca = Check for the best .cal file based on the event date                                        
        """)
    arg_parser.add_argument('meteors', \
        help="""Comma separated EMCCD events, e.g. "20190506_033100" or "20190506 03:31:00 UTC ..." .""")
    arg_parser.add_argument('--emccd', metavar='EMCCD', type=bool, default=True,\
        help='Gets the EMCCD data files, set True by default.')
    arg_parser.add_argument('--wide', metavar='WIDE', type=bool, default=True,\
        help='Gets the CAMO wide-field data files, set True by default but turn False if the OS is Windows as VidChop does not work.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Override the event list if there are events given in the command line arguments
    if cml_args.meteors:
        event_list = cml_args.meteors.split(',')


    # Format event names to a uniform format
    event_list = formatEventNames(event_list)

    print('Events:', event_list)

    ############################################################

    if cml_args.emccd:
        # Run event processing EMCCD
        processEMCCDEvents(event_list)

        for event_name in event_list:
            event_dir =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)
            event_dir_data =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)

            # parse event datetime once per event
            event_dt = get_event_datetime(event_name)

            os.chdir(event_dir)
            for file in os.listdir():
                if file.endswith('.tar'):

                    if '1F' in file:
                        station = '01F'
                        shutil.copy2(os.path.join(SKYFIT2_PATH_EMCCD, station, 'emccd01F.config'),
                                     event_dir_data+os.sep+'tavis_F')

                        cal_path = find_best_cal_file(station, event_dt)
                        if cal_path is not None:
                            shutil.copy2(cal_path, event_dir_data+os.sep+'tavis_F')
                        else:
                            error_track.addError(f'No .cal file found for the station {station} in :',
                                                 SKYFIT2_PATH_EMCCD)

                    elif '1G' in file:
                        station = '01G'
                        shutil.copy2(os.path.join(SKYFIT2_PATH_EMCCD, station, 'emccd01G.config'),
                                     event_dir_data+os.sep+'tavis_G')

                        cal_path = find_best_cal_file(station, event_dt)
                        if cal_path is not None:
                            shutil.copy2(cal_path, event_dir_data+os.sep+'tavis_G')
                        else:
                            error_track.addError(f'No .cal file found for the station {station} in :',
                                                 SKYFIT2_PATH_EMCCD)

                    elif '2F' in file:
                        station = '02F'
                        shutil.copy2(os.path.join(SKYFIT2_PATH_EMCCD, station, 'emccd02F.config'),
                                     event_dir_data+os.sep+'elgin_F')

                        cal_path = find_best_cal_file(station, event_dt)
                        if cal_path is not None:
                            shutil.copy2(cal_path, event_dir_data+os.sep+'elgin_F')
                        else:
                            error_track.addError(f'No .cal file found for the station {station} in :',
                                                 SKYFIT2_PATH_EMCCD)

                    elif '2G' in file:
                        station = '02G'
                        shutil.copy2(os.path.join(SKYFIT2_PATH_EMCCD, station, 'emccd02G.config'),
                                     event_dir_data+os.sep+'elgin_G')

                        cal_path = find_best_cal_file(station, event_dt)
                        if cal_path is not None:
                            shutil.copy2(cal_path, event_dir_data+os.sep+'elgin_G')
                        else:
                            error_track.addError(f'No .cal file found for the station {station} in :',
                                                 SKYFIT2_PATH_EMCCD)

                    else:
                        error_track.addError('No .config and .cal file found for the station in :',
                                             SKYFIT2_PATH_EMCCD)

                    os.remove(file)
                    print('Removed ' + file)

            os.chdir('..')


    ############################################################

    #Check the OS if it is not Liux avoid getting the Klingon files
    if platform.system() != 'Linux':
        error_track.addError('The Klingon files can only be processed on a Linux machine with vidchop installed')
        cml_args.wide = False

    if cml_args.wide:

        # Run event processing Klingon
        processKlingonEvents(event_list)

        # remove the .bz2 files
        for event_name in event_list:
            event_dir =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)
            os.chdir(event_dir)
            for file in os.listdir():

                # if is a .bz2 file, remove it
                if file.endswith('.bz2'):
                    # if in the name of the file there is a 1F or 1G or 2F or 2G
                    if '1K' in file:
                        # copy the all the files from /srv/meteor/reductions/emccd/skyfit_plates/01F to event_dir+os.sep+'elgin_F'
                        shutil.copy2(os.path.join(SKYFIT2_PATH_KLINGON, '01K.config'), event_dir+os.sep+'tavis_K')
                        shutil.copy2(os.path.join(SKYFIT2_PATH_KLINGON, 'platepar_01K.cal'), event_dir+os.sep+'tavis_K')
                        # shutil.copytree('/srv/meteor/reductions/emccd/skyfit_plates/01F/', event_dir+os.sep+'tavis_F/')
                    elif '2K' in file:
                        shutil.copy2(os.path.join(SKYFIT2_PATH_KLINGON, '02K.config'), event_dir+os.sep+'elgin_K')
                        shutil.copy2(os.path.join(SKYFIT2_PATH_KLINGON, 'platepar_02K.cal'), event_dir+os.sep+'elgin_K')
                        # shutil.copytree('/srv/meteor/reductions/emccd/skyfit_plates/02F/', event_dir+os.sep+'elgin_F/')
                        
                    else:
                        error_track.addError('No .config and .cal file found for the station in :',SKYFIT2_PATH_KLINGON)
                    os.remove(file)
                    print('Removed ' + file)
            # go back to the original directory 20220729_044924,20220726_070832,20210804_052106,20210802_055939,20200726_060419
            os.chdir('..')

    ############################################################

    # Check for errors at the end, to warn the used that things did not go perfectly fine
    print('---------------------')
    if error_track.errors:
        os.chdir(WORK_DIR)
        # check if there is no file called error.txt, if not create it
        if not os.path.isfile('errors_Skyfit2Prepare.txt'):
            open('errors_Skyfit2Prepare.txt', 'w').close()
        
        print('The following errors occured:')

        for error_msg in error_track.errors:
            print(error_msg)
            # print the errors to the error.txt file
            with open('errors_Skyfit2Prepare.txt', 'a') as f:
                f.write(error_msg + '\n')

    else:
        print('Successfully done!')






    

