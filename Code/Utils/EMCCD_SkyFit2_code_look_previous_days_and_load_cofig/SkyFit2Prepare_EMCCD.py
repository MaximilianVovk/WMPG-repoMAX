""" Collecting and organizing EMCCD dump files and calibration plates. """

from __future__ import print_function, absolute_import, division

import os
import errno
import glob
import shutil
import argparse
import subprocess
import tarfile
import datetime

# Version number
__version__ = 1.0

# Local working directory
WORK_DIR = os.getcwd()


# Data path variables
DATA_PATH  = '/srv/meteor/emccd/events/'
FLATS_PATH = '/srv/meteor/emccd/calib/auto-flats/'
BIAS_PATH  = '/srv/meteor/emccd/calib/auto-bias/'
PLATES_PATH = '/srv/meteor/emccd/plcheck/'
SKYFIT2_PATH = '/srv/meteor/reductions/emccd/skyfit_plates/'
PLATES_SUFFIX = '_plate.ast'
PLATE_FIRST = 'first.ast'
FLAT_FILE_NAME = 'flat.png'
BIAS_FILE_NAME = 'bias.png'
WORK_DIR_SUFFIX = '_emccd_skyfit2'
SITES = [['tavis_F', '1', 'F'], ['elgin_F', '2', 'F'],
        ['tavis_G', '1', 'G'], ['elgin_G', '2', 'G']]



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



def tarExtract(source, tar_file, site_dir):
    """Extracts files from tar'd directory. """
    
    tar_file_name = os.path.join(source, tar_file)
    tar_dir_name  = tar_file_name.strip('.tar')

    # Unpack the dump files from tar
    print('Unpacking', tar_file_name)
    tar_data = tarfile.open(tar_file_name)
    tar_data.extractall(path=source)
    tar_data.close()
    
    # Clean up extracted directory
    for file in os.listdir(tar_dir_name):
        file_path = os.path.join(tar_dir_name, file)
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



def copyFlats(night_name, site_id, destination_dir):
    """ Checks for existence of flats and copies them to appropriate site directory. """

    flats_night_path = os.path.join(FLATS_PATH, night_name)

    # Generate the flat image regex
    flat_regex = os.path.join(flats_night_path, 'flat*' + night_name + '*' + str(site_id) + '.png')

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
    
    
    
def copyBias(night_name, site_id, destination_dir):
    """ Checks for existence of flats and copies them to appropriate site directory. """

    bias_night_path = os.path.join(BIAS_PATH, night_name)

    # Generate the flat image regex
    bias_regex = os.path.join(bias_night_path, 'bias*' + night_name + '*' + str(site_id) + '.png')

    # Check if such a file exists
    bias_found = glob.glob(bias_regex)

    # If the flats were found, copy them to the local directory
    if bias_found:

        bias_name = bias_found[0]

        # Copy the flat file to the site directory
        shutil.copy2(os.path.join(bias_night_path, bias_name), os.path.join(destination_dir, BIAS_FILE_NAME))

        return True

    else:
        return False
    


def copyPlates(night_name, site_id, destination_dir):
    """ Checks for existence of plates and copies them to appropriate site directory. """

    plates_night_path = os.path.join(PLATES_PATH, night_name)

    # Generate the plate image regex
    plate_regex = os.path.join(plates_night_path, 'fit*' + night_name + '*' + str(site_id) + '.ast')

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



def processEMCCDEvents(event_list):
    """ Creates the folder structure for METAL, and fetches the data and calibration files. """

    for event_name in event_list:
        
        # Create a directory for the event
        event_dir =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)

        # Make the event directory
        mkdirP(event_dir)


        # Get the name of the night of the event
        night_name = extractNights([event_name])[0]

        # Regex for all CAMO widefiled files for this event
        files_regex = 'ev_' + event_name + '*.tar'
        file_regex_dir = os.path.join(DATA_PATH, night_name, files_regex)

        print('Copying files :', file_regex_dir)

        # Copy files of the event from all sites
        copied_files = regexCopy(file_regex_dir, os.path.join(event_dir))

        print('Copied files:')
        for cp_file_name in copied_files:
            print('  ' + cp_file_name)

        # Go through all sites
        for site_name, site_num, cam_code in SITES:

            # Make a site directory
            site_dir = os.path.join(event_dir, site_name)
            mkdirP(site_dir)

            event_site_file_found = False

            # Find the appropriate raw file to use for vidchop
            for event_site_file_name in os.listdir(event_dir):

                # Check if this is the file from the given station
                if (event_name in event_site_file_name) and (str(site_num) \
                    + cam_code + '.tar' in event_site_file_name):

                    event_site_file_found = True
                    break

            # Check if the file was found
            if not event_site_file_found:
                error_track.addError(event_name +': WARNING! Data file for site: ' + site_name + ' not found!')
                os.rmdir(site_dir)
                continue

            else:
                print('Data file for site: ' + site_name + ' successfully copied!')



            # Extract dump and plate files from tar
            tarExtract(event_dir, event_site_file_name, site_dir)

            # Copy flat image
            copy_flat_status = copyFlats(night_name, site_num+cam_code, site_dir)

            if not copy_flat_status:
                error_track.addError(event_name +': WARNING! Flat file for site: ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_flat_status = copyFlats(previous_night_name, site_num+cam_code, site_dir)
                    if copy_flat_status:
                        error_track.addError(event_name +': SOLVED! Found valid Flat for ' + site_name + ' in : ' +FLATS_PATH+''+ previous_night_name)
                        break
                    if previous_day==29:
                        error_track.addError(event_name +': No Flat found for ' + site_name + ' in the last 30 days, try look in : '+FLATS_PATH+''+ night_name )

            else:
                print('Flat file for site: ' + site_name + ' successfully copied!')


            # Copy bias image
            copy_bias_status = copyBias(night_name, site_num+cam_code, site_dir)

            if not copy_bias_status:
                error_track.addError(event_name +': WARNING! Bias file for site: ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_bias_status = copyBias(previous_night_name, site_num+cam_code, site_dir)
                    if copy_bias_status:
                        error_track.addError(event_name +': SOLVED! Found valid Bias for ' + site_name + ' in : '+BIAS_PATH+''+ previous_night_name)
                        break
                    if previous_day==29:
                        error_track.addError(event_name +': No Bias found for ' + site_name + ' in the last 30 days, try look in : '+BIAS_PATH+''+ night_name )

            else:
                print('Bias file for site: ' + site_name + ' successfully copied!')
                

            # Copy the plates
            copy_plates_status = copyPlates(night_name, site_num+cam_code, site_dir)
            
            if not copy_plates_status:
                error_track.addError(event_name +': WARNING! Plate for site ' + site_name + ' not found!')
                # check the previous day up to 30 days before
                for previous_day in range(1,30):
                    format = '%Y%m%d'
                    # convert from string format to datetime format subtrct the number of days and convert back to string format
                    previous_night_name = (datetime.datetime.strptime(night_name, format)-datetime.timedelta(days=previous_day)).date().strftime(format)
                    copy_plates_status = copyPlates(previous_night_name, site_num+cam_code, site_dir)
                    if copy_plates_status:
                        error_track.addError(event_name +': SOLVED! Found valid Plate for ' + site_name + ' in : '+PLATES_PATH+''+ previous_night_name)
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
    arg_parser = argparse.ArgumentParser(description="""An automated script for
        collecting EMCCD dump and calibration files. 
        Author: Denis Vida, e-mail: dvida@uwo.ca, (c) 2017. 
        Edited 2023: Peter Quigley, email: pquigle@uwo.ca.
        Edited 2023-04-25: Maximilian Vovk, email: mvovk@uwo.ca = now the code look for previos days bias flat and plate and rename plate.ast as first.ast if there is no first.ast file and delete .tar files
        Edited 2023-09-19: Maximilian Vovk, email: mvovk@uwo.ca = add skyfit2 .config and .cal to the event directory                                 
        """)
    arg_parser.add_argument('meteors', \
        help="""Comma separated EMCCD events, e.g. "20190506 03:31:00 UTC ..." .""")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Override the event list if there are events given in the command line arguments
    if cml_args.meteors:
        event_list = cml_args.meteors.split(',')


    # Format event names to a uniform format
    event_list = formatEventNames(event_list)

    print('Events:', event_list)


    # Run event processing
    processEMCCDEvents(event_list)

    for event_name in event_list:
        event_dir =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)
        event_dir_data =  os.path.join(WORK_DIR, event_name + WORK_DIR_SUFFIX)
        # if there is no first.ast change the name of plate.ast to first.ast and remove the .tar files
        os.chdir(event_dir)
        for file in os.listdir():
            # if is a folder go into it
            if os.path.isdir(file):
                os.chdir(file)
                # count the number of .ast files
                num_ast = 0
                for file in os.listdir():
                    if file.endswith('.ast'):
                        num_ast += 1
                # if there is one .ast file it must be the plate.ast file
                if num_ast == 1:
                    os.rename('plate.ast','first.ast')
                    print('NO first! Change the name of plate.ast to first.ast') 
                os.chdir('..')
            # print(event_dir_data)
            # if is a .tar file, remove it
            if file.endswith('.tar'):
                # if in the name of the file there is a 1F or 1G or 2F or 2G
                if '1F' in file:
                    # copy the all the files from /srv/meteor/reductions/emccd/skyfit_plates/01F to event_dir+'/elgin_F'
                    shutil.copy2(os.path.join(SKYFIT2_PATH, '01F/emccd01F.config'), event_dir_data+'/tavis_F')
                    shutil.copy2(os.path.join(SKYFIT2_PATH, '01F/platepar_cmn2010.cal'), event_dir_data+'/tavis_F')
                    # shutil.copytree('/srv/meteor/reductions/emccd/skyfit_plates/01F/', event_dir+'/tavis_F/')
                elif '1G' in file:
                    shutil.copy2(os.path.join(SKYFIT2_PATH, '01G/emccd01G.config'), event_dir_data+'/tavis_G')
                    shutil.copy2(os.path.join(SKYFIT2_PATH, '01G/platepar_cmn2010.cal'), event_dir_data+'/tavis_G')
                    # shutil.copytree('/srv/meteor/reductions/emccd/skyfit_plates/01G/', event_dir+'/tavis_G/')
                elif '2F' in file:
                    shutil.copy2(os.path.join(SKYFIT2_PATH, '02F/emccd02F.config'), event_dir_data+'/elgin_F')
                    shutil.copy2(os.path.join(SKYFIT2_PATH, '02F/platepar_cmn2010.cal'), event_dir_data+'/elgin_F')
                    # shutil.copytree('/srv/meteor/reductions/emccd/skyfit_plates/02F/', event_dir+'/elgin_F/')
                elif '2G' in file:
                    shutil.copy2(os.path.join(SKYFIT2_PATH, '02G/emccd02G.config'), event_dir_data+'/elgin_G')
                    shutil.copy2(os.path.join(SKYFIT2_PATH, '02G/platepar_cmn2010.cal'), event_dir_data+'/elgin_G')
                    # shutil.copytree('/srv/meteor/reductions/emccd/skyfit_plates/02G/', event_dir+'/elgin_G/')
                else:
                    error_track.addError('No .config and .cal file found for the station in : /srv/meteor/reductions/emccd/skyfit_plates')

                os.remove(file)
                print('Removed ' + file)
        # go back to the original directory
        os.chdir('..')



    # Check for errors at the end, to warn the used that things did not go perfectly fine
    print('---------------------')
    if error_track.errors:
        os.chdir(WORK_DIR)
        # check if there is no file called error.txt, if not create it
        if not os.path.isfile('errors_MetalPrepare_EMCCD.txt'):
            open('errors_MetalPrepare_EMCCD.txt', 'w').close()
        
        print('The following errors occured:')

        for error_msg in error_track.errors:
            print(error_msg)
            # print the errors to the error.txt file
            with open('errors_MetalPrepare_EMCCD.txt', 'a') as f:
                f.write(error_msg + '\n')


    else:
        print('Successfully done!')






    

