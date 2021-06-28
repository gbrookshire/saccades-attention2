"""
Test the analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import load_data
import fixation_events
import dist_convert as dc


plt.ion()  # Interactive plots

n = 0  # Which subject to analyze

d = load_data.load_data(n)  # Load the data


# Plot the locations of the fixations
plt.figure()
plt.plot(d['eye'].fixations['x_avg'],
         d['eye'].fixations['y_avg'],
         'ob', alpha=0.4,
         markerfacecolor='none',
         label='Fixations')
plt.plot(*np.transpose(fixation_events.stim_locs),
         '*r',
         label='Stimulus centers')
plt.xlabel('X position')
plt.xlabel('Y position')
plt.xlim(0, dc.screen_res[0])
plt.ylim(0, dc.screen_res[1])
plt.legend()


# Check whether we see a visual potential at stimulus onset
d['raw'].load_data()  # Apply ICA
d['raw'].filter(0.1, 40)  # Bandpass filter
d['ica'].apply(d['raw'])
epochs = mne.Epochs(d['raw'],
                    d['meg_events'],
                    event_id=load_data.expt_info['event_dict']['stimuli'],
                    tmin=-0.2, tmax=1.0,
                    baseline=(None, 0))
evoked = epochs.average()
evoked.plot(spatial_colors=True)


# Check whether we see a fixation-induced potential
epochs = mne.Epochs(d['raw'],
                    d['fix_events'],
                    event_id=load_data.expt_info['event_dict']['fix_on'],
                    tmin=-0.2, tmax=1.0,
                    baseline=(None, 0))
evoked = epochs.average()
evoked.plot(spatial_colors=True)
