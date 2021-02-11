The file MEG_decoding_RDMs.mat contains a 5-dimensional matrix of decoding accuracies with dimensions:
16 (subjects) * sessions (2) * time points (1301, from -100 to *1200ms wrt to stimulus onset (at 0ms)) * conditions (92) * conditions (92)
The last 2 dimensions form represeentational dissimilarity matrices, symmetric across the diagonal, with the diagonal undefined (NaN).



The file visual_stimuli.mat contains a 92 dimensional struct with several fields
category                        :   string indicating category
human, face, animal, natural    :   scalar, indicating membership (0= not a member, 1= member)
pixel_values                    :   the actual images