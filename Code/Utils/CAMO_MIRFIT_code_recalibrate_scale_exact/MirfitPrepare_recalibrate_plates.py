from __future__ import print_function, absolute_import, division

import subprocess  
import os
import re
import numpy as np
import os
import errno
import glob
import shutil
import argparse
import bz2
import datetime

# Data path variables
WORK_DIR = os.getcwd()
PLATES_DIR = 'plates_'
LIST_TXT = [ 'Tavistock_scale_command.txt', 'Tavistock_exact_command.txt', 'Elginfield_scale_command.txt','Elginfield_exact_command.txt']
LIST_OUTPUT=['scale_01.aff', 'exact_01.ast', 'scale_02.aff','exact_02.aff']
MIR_SUF='_mir'

PERCENTILE_BAD_STAR=95 # go faster if decreesed but might stop too soon
MEAN_STAR=0.1 # go faster if decreesed but less accurate
MIN_STAR_COUNT = 15 # if smaller than 10 might the output file might not work

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

        # Copy the file to destination if already exists
        shutil.copy2(file_path, os.path.join(destination, file_name), follow_symlinks=True)


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


def mircalibrate(comand_mircal):
    # run the command in the terminal
    # but avoid OSError: [Errno 36] File name too long: and save the output in a variable
    output_mir = subprocess.check_output(['bash', '-c', comand_mircal])
    # save output_mir as a string
    output_mir= str(output_mir)
    sv_nm=[]
    first_line = output_mir.split('\\n')[0]
    # go to the next line
    output_mir = output_mir.split('\\n', 1)[1]
    # while there are \n in output_mir create a variable with the first line of the output
    while '\\n' in output_mir:
        first_line = output_mir.split('\\n')[0]
        # go to the next line
        output_mir = output_mir.split('\\n', 1)[1]
        # find the numbers in the string and save them in a list
        numbers_mir = re.findall(r"[-+]?\d*\.\d+|\d+", first_line)
        # convert the numbers to floats
        numbers_mir = [float(i) for i in numbers_mir]
        # break if there are no numbers in the line
        if len(numbers_mir) == 0:
            break
        # create in a matrix the first and the last numbers
        sv_nm.append([numbers_mir[0], numbers_mir[-1]])

    sv_nm = np.array(sv_nm)
    # order the matrix by the last number
    sv_nm = sv_nm[sv_nm[:,1].argsort()]

    # find the mean of the last number
    mean_nm = np.mean(sv_nm[:,1])

    sv_nm_perc=[]
    # if the mean is bigger than 1
    if mean_nm > MEAN_STAR:
        # find the rows over the 95th percenteil of the last number
        sv_nm_perc = sv_nm[sv_nm[:,1] > np.percentile(sv_nm[:,1], PERCENTILE_BAD_STAR)]
        # add the first numbers of sv_nm_90 to comand_mircal

    if len(sv_nm)-len(sv_nm_perc) < MIN_STAR_COUNT:
        sv_nm_perc = []
    
    return sv_nm_perc

def processEvents(event_list):
    """ Creates the folder structure for METAL, and fetches the data and calibration files. """

    for event_name in event_list:
        # Get the name of the night of the event
        night_name = extractNights([event_name])[0]

        # Create a directory for the event
        event_dir =  os.path.join(WORK_DIR, PLATES_DIR + night_name)

        # go to the event directory
        os.chdir(event_dir)
        nm_of_txt=0
        for txt_name_comand in LIST_TXT:
            print('Check: ' + txt_name_comand + ' file in the folder')
            # Find the appropriate file to use
            for file in os.listdir():
                # if there is Elginfield_exact_command.txt
                if file == txt_name_comand:
                    print('Data of ' + txt_name_comand + ' successfully found!')
                    with open(file, 'r') as f:
                        lines = f.readlines()
                        # split lines at "b'exact"
                        for line in lines:
                            # check if the line contains the command before "b'" 
                            if "b'" in line:
                                comand_mircal = line.split("b'")[0]

            if not os.path.isfile(txt_name_comand):
                error_track.addError(event_name +': WARNING! ' + txt_name_comand + ' not found in '+ PLATES_DIR + night_name )
                continue     
            else:           
                if not comand_mircal:
                    error_track.addError(event_name +': WARNING! ' + txt_name_comand + ' in '+ PLATES_DIR + night_name +' empty or not correct!' )
                    continue
                else:
                    # while comand_mircal is not empty
                    while comand_mircal:
                        # run mircalibrate
                        bad_star = mircalibrate(comand_mircal)
                        # if sv_nm_perc is empty
                        if len(bad_star) == 0:
                            # break
                            break
                        # else
                        else:
                            # run mircalibrate again with new bad stars
                            for i in range(len(bad_star)):
                                comand_mircal = comand_mircal +','+ str(int(bad_star[i,0]))
                    print('The command of ' + txt_name_comand + ' with an average error of '+ str(MEAN_STAR) +' is: \n' + comand_mircal)
            os.chdir('..')
            # check if there is a folder called "mircal" 
            if os.path.isdir(event_name+MIR_SUF):
                file_regex_dir = os.path.join(WORK_DIR, event_name+MIR_SUF, LIST_OUTPUT[nm_of_txt])
                print('Copying '+ LIST_OUTPUT[nm_of_txt] +' from '+ PLATES_DIR + night_name +' to :', file_regex_dir)
                # Copy files of the event from all sites
                regexCopy(file_regex_dir, os.path.join(event_dir)) 
            os.chdir(event_dir)
            nm_of_txt=nm_of_txt+1  









if __name__ == "__main__":

    # Tracks errors during the running of the program
    error_track = ErrorTracker()


    # List of events to be processed
    event_list = []

    # Handle command line arguments
    arg_parser = argparse.ArgumentParser(description="""An automated script for recalibrate plates. 
        Author: Maximilian Vovk, e-mail: mvovk@uwo.ca, (c) 2023
        """)
    arg_parser.add_argument('meteors', \
        help="""Comma separated events, e.g. "20190506 03:31:00 UTC ..." .""")


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

    # Check for errors at the end, to warn the used that things did not go perfectly fine
    print('---------------------')
    if error_track.errors:
        os.chdir(WORK_DIR)
        # check if there is file called error.txt, if not create it
        if not os.path.isfile('errors_recalib_Mirfit.txt'):
            open('errors_recalib_Mirfit.txt', 'w').close()
        
        print('The following errors occured:')

        for error_msg in error_track.errors:
            print(error_msg)
            # print the errors to the error.txt file to the end of the file
            with open('errors_recalib_Mirfit.txt', 'a') as f:
                f.write(error_msg + '\n')


    else:
        print('Successfully done!')

