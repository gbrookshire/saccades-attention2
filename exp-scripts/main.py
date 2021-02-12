"""
Attention and preprocessing before a saccade
Spring 2021
G.Brookshire@bham.ac.uk
"""

"""
TODO
- What should we use as probe words?
- Update instructions
- Check for old/unused settings
- Sometimes a set of stimuli is not shown after 2 probe trials in a row
    - The trial after the first probe doesn't appear?
- Fix the problem that causes occasional crashing w/ duplicated stims
"""

# Standard libraries
import os
import datetime
import numpy as np
import scipy.io as sio
import yaml

# Psychopy
from psychopy import parallel
from psychopy import visual, core, data, event, monitors

# Custom modules
import refcheck
import dist_convert as dc
import eye_wrapper


############
# Settings #
############

# Set to False for testing without triggers, eye-tracker, etc
IN_MEG_LAB = False

FULL_SCREEN = False

START_TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
RT_CLOCK = core.Clock() # for measuring response times

TRIGGERS = {'response': 1,
            'fixation': 2,
            'stimuli': 4,
            'probe': 8,
            'drift_correct_start': 16,
            'drift_correct_end': 32}

KEYS = {'break': 'escape',
        'drift': 'return',
        'accept': 'space',
        'yes': '7',
        'no': '8'}

COLORS = {'cs': 'rgb', # ColorSpace
          'white': [1.0, 1.0, 1.0],
          'grey': [0.0, 0.0, 0.0],
          'black': [-1.0, -1.0, -1.0],
          'pink': [1.0, -0.9, -0.1]}

if FULL_SCREEN:
    SCREEN_RES = [1920, 1080] # Full res on the Propixx projector
else:
    SCREEN_RES = [1000, 1000]
SCREEN_CENTER = [0, 0] #[int(z / 2) for z in SCREEN_RES]
STIM_SIZE_DEG = 2.0 # Stim size in visual degrees
STIM_DIST_DEG = 5.0 # Distance b/w centers of stimuli in vis deg
STIM_SIZE = int(dc.deg2pix(STIM_SIZE_DEG)) # Size in pixels
STIM_DIST = int(dc.deg2pix(STIM_DIST_DEG)) # Distance b/w stimuli
STIM_DUR = 2.0 # Duration the stimuli stay on screen
RESPONSE_CUTOFF = 4.0 # Respond within this time
ITI = 0.2 # Inter-trial interval
FIX_DUR = (0.5, 1.0) # Hold fixation for X seconds before starting trial
FIX_THRESH_DEG = 1.0 # Subject must fixate w/in this distance to start trial
FIX_THRESH = int(dc.deg2pix(FIX_THRESH_DEG))
P_PROBE = 0.5 # Probability of getting a word probe on this trial
N_REPS_PER_LOC = 2 # How many times each object appears in each location
BLOCK_LENGTH = 25 # Number of trials per block

END_EXPERIMENT = 9999 # Numeric tag signals stopping expt early

LOG_DIR = '../logfiles/'
assert os.path.exists(LOG_DIR), 'Logfile directory does not exist'

STIM_DIR = '../stimuli/'
assert os.path.exists(STIM_DIR), 'Stimuli directory does not exist'

# Load instructions
with open('instruct.txt') as f:
    instruct_text = f.readlines()

# Load the list of probe words
with open('probes.txt') as f:
    probe_words = [w.strip() for w in f.readlines()]

# Load info about which images to use
with open(f"stimuli.yaml") as f:
    stim_info = yaml.load(f, Loader=yaml.SafeLoader)
stims_to_show = []
for stims in stim_info.values():
    stims_to_show.extend(stims)
stims_to_show = np.array(stims_to_show)

# Initialize external equipment
if IN_MEG_LAB:
    refresh_rate = 120.0
    port = parallel.ParallelPort(address=0xBFF8)
    port.setData(0)
    el = eye_wrapper.SimpleEyelink(SCREEN_RES)
    el.startup()

    def eye_pos():
        """ Get the eye position
        """
        pos = el.el.getNewestSample()
        pos = pos.getRightEye()
        pos = pos.getGaze() # eye position in pix (origin: bottom right)
        return pos

    def send_trigger(trig):
        """ Send triggers to the MEG acquisition computer
        and the EyeLink computer.
        """
        t = TRIGGERS[trig]
        port.setData(t)
        el.trigger(t)

    def reset_port():
        """ Reset the parallel port to avoid overlapping triggers
        """
        wait_time = 0.003
        core.wait(wait_time)
        port.setData(0)
        core.wait(wait_time)

else: # Dummy functions for dry-runs on my office desktop
    refresh_rate = 60.0
    el = eye_wrapper.DummyEyelink()

    def eye_pos():
        pos = win_center
        pos = np.int64(dc.origin_psychopy2eyelink(pos))
        return pos

    def send_trigger(trig):
        print('Trigger: {}'.format(trig))

    def reset_port():
        pass


######################
# Window and Stimuli #
######################

win_center = (0, 0)

win = visual.Window(SCREEN_RES,
                    monitor='propixxMonitor',
                    fullscr=FULL_SCREEN,
                    color=COLORS['grey'], colorSpace=COLORS['cs'],
                    allowGUI=False)

# parameters used across stimuli
stim_params = {'win': win, 'units': 'pix'}
circle_params = {'fillColor': COLORS['white'],
                 'lineColor': COLORS['white'],
                 'fillColorSpace': COLORS['cs'],
                 'lineColorSpace': COLORS['cs'],
                 **stim_params}

text_stim = visual.TextStim(pos=win_center, text='hello', # For instructions
                            color=COLORS['white'], colorSpace=COLORS['cs'],
                            height=32,
                            **stim_params)

fixation = visual.Circle(radius=10, pos=win_center, **circle_params)
drift_fixation = visual.Circle(radius=5, pos=win_center, **circle_params)

# For marking the gaze position
eye_marker = visual.Circle(radius=20, pos=win_center, **circle_params)
eye_marker.fillColor = COLORS['pink']

# Make psychopy stimulus objects
pic_stims = {}
for n in stims_to_show:
    stim_fname = f'{STIM_DIR}{n}.jpg'
    s = visual.ImageStim(image=stim_fname,
                         size=(STIM_SIZE, STIM_SIZE),
                         colorSpace=COLORS['cs'],
                         **stim_params)
    pic_stims[n] = s


###################
# Make the trials #
###################

stim_locations = ('left', 'center', 'right')

# Make the lists of stimuli at each location
# Every image is shown once at each location before repeating the images
# Images are not duplicated within a trial
stim_list_by_loc = {loc: [] for loc in stim_locations}
for i_rep in range(N_REPS_PER_LOC):
    stims_temp = {}
    for loc in stim_locations:
        stims_temp[loc] = stims_to_show.copy()
        np.random.shuffle(stims_temp[loc])
    # Check for duplicates within a trial
    i_iter = 0
    max_iter = 100
    while True:
        i_iter += 1
        if i_iter > max_iter:
            print("Couldn't make a version without duplicates")
            import sys; sys.exit()

        stims_by_trial = np.array([stims_temp[loc] for loc in stim_locations])
        n_unique = np.apply_along_axis(lambda x: len(set(x)),
                                       axis=0,
                                       arr=stims_by_trial)
        dups = np.nonzero(n_unique != len(stim_locations))[0] # inx of dups
        # Break out of this loop when there are no more duplicates
        if len(dups) == 0:
            break
        else:
            print('Fixing trials to avoid duplicates')
            # Shuffle any duplicates
            for inx in dups:
                s_l = stims_temp['left']
                s_c = stims_temp['center']
                s_r = stims_temp['right']
                # If left stim is duplicated switch it with the adjacent stim
                inx_switch_stim ### FIXME Find the next stim where moving this wouldn't make a problem
                if s_l[inx] in (s_c[inx], s_r[inx]):
                    s_l[inx - 1], s_l[inx] = s_l[inx], s_l[inx - 1]
                # Otherwise switch the center stim
                else:    
                    s_c[inx - 1], s_c[inx] = s_c[inx], s_c[inx - 1]
    # Add the stimulus lists to the trial order
    for loc in stim_locations:
        stim_list_by_loc[loc].extend(stims_temp[loc])

# Build the list of trial info dictionaries
choice = np.random.choice
trial_info = []
for s_left, s_center, s_right in zip(*stim_list_by_loc.values()):
    d = {}
    d['stim_left'] = s_left
    d['stim_center'] = s_center
    d['stim_right'] = s_right
    if choice([True, False], p=(P_PROBE, 1 - P_PROBE)):
        d['probe_word'] = choice(probe_words)
    else:
        d['probe_word'] = None
    d['fix_dur'] = np.random.uniform(*FIX_DUR)
    trial_info.append(d)

trials = data.TrialHandler(trial_info, nReps=1, method='sequential')


#########################
# Stimulus presentation #
#########################

def euc_dist(a, b):
    """ Euclidean distance between two (x,y) pairs
    """
    d = sum([(x1 - x2)**2 for x1,x2 in zip(a, b)]) ** (1/2)
    return d


def show_text(text):
    """ Show text at the center of the screen
    """
    text_stim.text = text
    text_stim.draw()
    win.flip()


def instructions(text):
    """ Show instructions and go on after pressing space
    """
    show_text(text)
    event.waitKeys(keyList=['space'])
    win.flip(clearBuffer=True) # clear the screen
    core.wait(0.2)


def drift_correct():
    """ Eye-tracker drift correction.
    Press SPACE on the Eyelink machine to accept the current position.
    """
    reset_port()
    core.wait(0.2)
    # Draw a fixation dot
    drift_fixation.draw()
    win.flip()
    send_trigger('drift_correct_start')
    reset_port()
    # Do the drift correction
    fix_pos = np.int64(dc.origin_psychopy2eyelink(drift_fixation.pos))
    el.drift_correct(fix_pos)
    send_trigger('drift_correct_end')
    reset_port()


def experimenter_control():
    """ Check for experimenter key-presses to pause/exit the experiment or
    correct drift in the eye-tracker.
    """
    r = event.getKeys(KEYS.values())
    if KEYS['break'] in r:
        show_text('End experiment? (y/n)')
        core.wait(1.0)
        event.clearEvents()
        r = event.waitKeys(keyList=['y', 'n'])
        if 'y' in r:
            return END_EXPERIMENT
    elif KEYS['drift'] in r:
        drift_correct()


def run_trial(trial):
    print(trial)

    reset_port()
    event.clearEvents()

    # Wait for fixation and check for experimenter input
    fixation.draw()
    win.flip()
    send_trigger('fixation')
    reset_port()
    t_fix = core.monotonicClock.getTime() # Start a timer
    core.wait(0.2)
    while True:
        # Check for experimenter control to end or correct drift
        if experimenter_control() == END_EXPERIMENT:
            return END_EXPERIMENT
        d = euc_dist(dc.origin_eyelink2psychopy(eye_pos()), win_center)
        t_now = core.monotonicClock.getTime()
        # Reset timer if not looking at fixation
        if (d > FIX_THRESH):
            t_fix = t_now
        # If they are looking at the fixation, and have looked long enough
        elif (t_now - t_fix) > trial['fix_dur']:
            break
        # If they are looking, but haven't held fixation long enough
        else:
            fixation.draw()
            win.flip()

    # Present the stimuli

    stim_left = pic_stims[trial['stim_left']]
    stim_left.pos = (SCREEN_CENTER[0] - STIM_DIST, SCREEN_CENTER[1])
    stim_left.draw()

    stim_center = pic_stims[trial['stim_center']]
    stim_center.pos = SCREEN_CENTER
    stim_center.draw()

    stim_right = pic_stims[trial['stim_right']]
    stim_right.pos = (SCREEN_CENTER[0] + STIM_DIST, SCREEN_CENTER[1])
    stim_right.draw()

    win.flip()
    send_trigger('stimuli')
    reset_port()
    core.wait(STIM_DUR)
    trials.addData('stim_onset', core.monotonicClock.getTime())
    
    # Show the probe
    if trial['probe_word'] != None:
        show_text(trial['probe_word'])
        send_trigger('probe')
        RT_CLOCK.reset()
        reset_port()

        # Wait for a key press
        event.clearEvents()
        r = event.waitKeys(maxWait=RESPONSE_CUTOFF,
                           keyList=[KEYS['yes'], KEYS['no']],
                           timeStamped=RT_CLOCK)
        if r is not None:
            send_trigger('response')
            reset_port()
            keypress, rt = r[0]
            trials.addData('resp', keypress)
            trials.addData('rt', rt)

    win.flip(clearBuffer=True)
    core.wait(ITI)

    return experimenter_control()


def eye_pos_check():
    """ Check whether the stimulus is following the eye position
    """
    event.clearEvents()
    while True:
        # Check for control keypreses
        r = event.getKeys(KEYS.values())
        if KEYS['break'] in r:
            break
        elif KEYS['accept'] in r:
            drift_correct()

        pos = eye_pos()
        pos = dc.origin_eyelink2psychopy(pos)
        eye_marker.pos = pos # Mark the current fixation

        # Draw the stimuli
        for s in eye_testers:
            s.draw()
        eye_marker.draw()
        win.flip()


def run_exp():
    """ Coordinate the different parts of the experiment
    """

    # A few tests before beginning the experiment
    refcheck.check_refresh_rate(win, refresh_rate)
    # eye_pos_check()

    # Instructions
    for line in instruct_text:
        instructions(line)

    # Run the trials
    for i_trial, trial in enumerate(trials):
        status = run_trial(trial)
        if status == END_EXPERIMENT:
            break
        # Prompt to take a break between blocks
        if (i_trial > 0) and (i_trial % BLOCK_LENGTH == 0):
            msg = "Take a brief break and let the experimenter " \
                  "know when you're ready to go on."
            show_text(msg)
            event.waitKeys(keyList=['return'], maxWait=9999)
            show_text('Ready?')
            event.waitKeys(keyList=['return'], maxWait=9999)
            drift_correct()

    # Save the data
    fname = '{}/{}.csv'.format(LOG_DIR, START_TIME)
    trials.saveAsWideText(fname, encoding='ASCII',
                          delim=';', fileCollisionMethod='rename')

    show_text('That was it -- thanks!')
    event.waitKeys(keyList=['escape'], maxWait=30)

    # Close everything down
    win.close()
    if IN_MEG_LAB:
        el.shutdown()
    core.quit()

