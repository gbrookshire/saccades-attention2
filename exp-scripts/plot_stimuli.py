"""
Plot the images used in this study
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

STIM_DIR = '../stimuli/'
assert os.path.exists(STIM_DIR), 'Stimuli directory does not exist'

# Load info about which images to use
with open(f"stimuli.yaml") as f:
    stim_info = yaml.load(f, Loader=yaml.SafeLoader)
stims_to_show = []
for stims in stim_info.values():
    stims_to_show.extend(stims)
stims_to_show = np.array(stims_to_show)

plt.clf()
for i_plot, i_stim in enumerate(stims_to_show):
    img_fname = f'{STIM_DIR}{i_stim}.jpg'
    img = plt.imread(img_fname)
    plt.subplot(7, 7, i_plot + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(i_stim)

plt.tight_layout()
plt.savefig(f'{STIM_DIR}stim_selection.png')
