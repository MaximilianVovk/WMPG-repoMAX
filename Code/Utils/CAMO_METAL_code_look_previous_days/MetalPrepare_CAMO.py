""" Collecting and organizing CAMO widefield .vid files and calibration plates. """

from __future__ import print_function, absolute_import, division

import os
import errno
import glob
import shutil
import argparse
import subprocess
import bz2
import datetime


# Version number
__version__ = 1.0

# Local working directory
WORK_DIR = os.getcwd()


# Data path variables
DATA_PATH = '/srv/meteor/klingon/events/'
FLATS_PATH = '/srv/meteor/klingon/flats/'
PLATES_PATH = '/srv/meteor/klingon/plcheck/' 
PLATE_FIRST = 'first.ast'
FLAT_FILE_NAME = 'flat.png'
WORK_DIR_SUFFIX = '_klingon_met'
SITES = [['tavis', 1], ['elgin', 2]]



class ErrorTracker(object):
    """ Indicates if any errors or warnings occured during the running of the program. """

    def __init__(self):

        self.errors = []


    def addError(self, error_msg):

        self.errors.append(error_msg)
        print(error_msg)



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



def copyFlats(night_name, site_id, destination_dir):
    """ Checks for existence of flats and copies them to appropriate site directory. """

    flats_night_path = os.path.join(FLATS_PATH, night_name)

    # Generate the flat image regex
    flat_regex = os.path.join(flats_night_path, 'flat*' + night_name + '*' + str(site_id) + 'K.png')

    # Check if such a file exists
    flats_found = glob.glob(flat_regex)

    # If the flats were found, copy them to the local directory
    if flats_found:

        flat_name = flats_found[0]

        # Copy the flat file to the site directory
        shutil.copy2(os.path.join(flats_night_path, flat_name), os.path.join(destination_dir, FLAT_FILE_NAME))

        return True

    else:
        return False
    


def copyPlates(night_name, site_id, destination_dir):
    """ Checks for existence of plates and copies them to appropriate site directory. """

    plates_night_path = os.path.join(PLATES_PATH, night_name)

    # Generate the plate image regex
    plate_regex = os.path.join(plates_night_path, 'fit*' + night_name + '*' + str(site_id) + 'K.ast') # fit_20210523_02K.ast

    # Check if such a file exists
    plates_found = glob.glob(plate_regex)

    # If the plates were found, copy them to the local directory
    if plates_found:

        plate_name = plates_found[0]

        # Copy the plate file to the site directory
        shutil.copy2(os.path.join(plates_night_path, plate_name), os.path.join(destination_dir, PLATE_FIRST))

        return True

    else:
        return False



def processEvents(event_list):
    """ Creates the folder structure for METAL, and fetches the data and calibration files. """

    for event_name in event_list:
        
        # Create a directory for the event
        event_dir =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)

        # Make the event directory
        mkdirP(event_dir)


        # Get the name of the night of the event
        night_name = extractNights([event_name])[0]

        # Regex for all CAMO widefiled files for this event
        files_regex = 'ev_' + event_name + '*K*.vid.bz2'
        file_regex_dir = os.path.join(DATA_PATH, night_name, files_regex)

        print('Copying files :', file_regex_dir)

        # Copy files of the event from all sites
        copied_files = regexCopy(file_regex_dir, os.path.join(event_dir))

        print('Copied files:')
        for cp_file_name in copied_files:
            print('  ' + cp_file_name)

        # Go through all sites
        for site_name, site_id in SITES:

            # Make a site directory
            site_dir = os.path.join(event_dir, site_name)
            mkdirP(site_dir)

            event_site_file_found = False

            # Find the appropriate raw file to use for vidchop
            for event_site_file_name in os.listdir(event_dir):

                # Check if this is the file from the given station
                if (event_name in event_site_file_name) and (str(site_id) \
                    + 'K.vid.bz2' in event_site_file_name):

                    event_site_file_found = True
                    break

            # Check if the file was found
            if not event_site_file_found:
                error_track.addError(event_name +': WARNING! Data file for site: ' + site_name + ' not found!')
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
            copy_flat_status = copyFlats(night_name, site_id, site_dir)

            if not copy_flat_status:
                error_track.addError(event_name +': WARNING! Flat file for site: ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_flat_status = copyFlats(previous_night_name, site_id, site_dir)
                    if copy_flat_status:
                        error_track.addError(event_name +': SOLVED! Found valid Flat for ' + site_name + ' in : ' +FLATS_PATH+''+ previous_night_name )
                        break
                    if previous_day==29:
                        error_track.addError(event_name +': No Flat found for ' + site_name + ' in the last 30 days, try look in : '+FLATS_PATH+''+ night_name )

            else:
                print('Flat file for site: ' + site_name + ' successfully copied!')


            # Copy the plates
            copy_plates_status = copyPlates(night_name, site_id, site_dir)
            
            if not copy_plates_status:
                error_track.addError(event_name +': WARNING! Plate for site ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_plates_status = copyPlates(previous_night_name, site_id, site_dir)
                    if copy_plates_status:
                        error_track.addError(event_name +': SOLVED! Found valid Plate for ' + site_name + ' in : ' +PLATES_PATH+''+ previous_night_name )
                        break
                    if previous_day==29:
                        error_track.addError(event_name +': No Plate found for ' + site_name + ' in the last 30 days, try look in : '+PLATES_PATH+''+ night_name )

            else:
                print('Plate for site: ' + site_name + ' successfully copied!')








if __name__ == "__main__":

    # Tracks errors during the running of the program
    error_track = ErrorTracker()


    # List of events to be processed
    event_list = []

    # Handle command line arguments
    arg_parser = argparse.ArgumentParser(description="""An automated script for collecting CAMO .vid and 
        calibration files. 
        Author: Denis Vida, e-mail: dvida@uwo.ca, (c) 2017
        Edited 2023-04-25: Maximilian Vovk, email: mvovk@uwo.ca = now the code look for previos days flat and plate and delete .bz2 files
        """)
    arg_parser.add_argument('meteors', \
        help="""Comma separated CAMO events, e.g. "20190506 03:31:00 UTC ..." .""")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Override the event list if there are events given in the command line arguments
    if cml_args.meteors:
        event_list = cml_args.meteors.split(',')


    # Format event names to a uniform format
    event_list = formatEventNames(event_list)

    print('Events:', event_list)


    # Run event processing
    processEvents(event_list)

    # remove the .bz2 files
    for event_name in event_list:
        event_dir =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)
        os.chdir(event_dir)
        for file in os.listdir():
            # if is a .bz2 file, remove it
            if file.endswith('.bz2'):
                os.remove(file)
                print('Removed ' + file)
        # go back to the original directory
        os.chdir('..')

    # Check for errors at the end, to warn the used that things did not go perfectly fine
    print('---------------------')
    if error_track.errors:
        os.chdir(WORK_DIR)
        # check if there is no file called error.txt, if not create it
        if not os.path.isfile('errors_MetalPrepare_CAMO.txt'):
            open('errors_MetalPrepare_CAMO.txt', 'w').close()
        
        print('The following errors occured:')

        for error_msg in error_track.errors:
            print(error_msg)
            # print the errors to the error.txt file
            with open('errors_MetalPrepare_CAMO.txt', 'a') as f:
                f.write(error_msg + '\n')


    else:
        print('Successfully done!')



