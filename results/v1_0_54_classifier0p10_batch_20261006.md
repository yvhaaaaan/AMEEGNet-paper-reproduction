# v1.0.54 classifier max-norm audit

The classifier-row max-norm threshold `0.10` was selected only as a
source-session validation candidate. The follow-up nine-subject batch used
the full 288-trial source session, 1000 epochs, and one final target-session
evaluation, with all other settings fixed to the current candidate.

The final-only result was **65.63% +/- 22.42%** (sample SD, n=9), or
-15.55 percentage points from the paper's 81.17% mean. All nine records pass
the mechanical audit, but the training curves show late instability for
several subjects and the configuration is rejected. The classifier threshold
`0.25` remains the retained value.
