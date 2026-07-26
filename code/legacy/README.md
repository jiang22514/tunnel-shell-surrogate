# Extended analyses not included in the compact release

The compact release contains everything needed to reproduce the primary
20-section profile metrics from the archived arrays and pretrained weights.

The following extended analyses require additional inputs and are therefore
not presented as runnable entry points here:

- the 2D full-field baseline, which requires registered displacement and
  temperature fields;
- the surveyed Genieshan case, which requires the surveyed outline and its
  finite-element reference export;
- intermediate statistical scripts tied to those extended outputs; and
- an earlier C30 eccentric-compression utility that is unrelated to the
  paper's reported C40 cracking-extent check.

Earlier author-workspace scripts remain recoverable from the repository
history. The supported entry points are `../reproduce_headline.py`,
`../train_mechanics.py`, and `../train_temperature.py`.
