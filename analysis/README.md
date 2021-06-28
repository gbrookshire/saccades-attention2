## Data organization

Within the `data/` directory, there should be a few sub-directories:
- `raw`: Raw MEG data
- `ica`: ICA files, saved by `artifacts.py`
- `annotations`: Manual artifact tags
- `logfiles`: Behavioral logfiles
- `eyelink`: Data from the eye-tracker
    - This will have each subject's data, as well as a directory called `ascii` that holds the converted ASCII-format eye-tracker data.

The `data` directory should also have the file `subject_info.csv`.

## Converting Eye-tracker data

To convert the eye-tracker data to a useable ASCII format, run the file `eyelink_ascii.sh`.

## Identifying artifacts

Identify artifacts for each subject by running `python artifacts.py` in the terminal, and then entering the subject snumber from `subject_info.csv`. Alternatively, you can `import artifacts` in python, and then run `artifacts.identify_artifacts(n)`, where `n` is the subject number.

Out-of-trial is highlighted in red
X out of the artifact browser window
ICA: click on trace to mark as bad
