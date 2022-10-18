# saccades-attention2

How does the brain gather information before an eye movement? (Take 2)

This repository holds the code used for an experiment I did at the University of Birmingham in Ole Jensen's lab. We didn't end up publishing anything from this study.


# Design

- Procedure
    - Central fixation point 
        - 0.5 to 1.5 s
    - Three images arranged horizontally on the screen
        - 2 s
    - On 10% of trials, probe processing
        - Show a word. Participants respond (y/n) whether that object was present in the display.
- Stimuli
    - 48 stimuli from Cichy et al (2014)
    - Each image 2 deg wide
    - Image centers 5 deg apart


# Requirements

## Stimulus presentation
- [Psychopy 3.2.3](https://www.psychopy.org/download.html)

## Analysis
- [mne-python 0.19.2](https://mne.tools/stable/install/mne_python.html) 
