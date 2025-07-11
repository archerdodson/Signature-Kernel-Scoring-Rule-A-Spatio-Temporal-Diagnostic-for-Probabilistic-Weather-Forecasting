This repository contains three sections, Model Training, Deterministic Scorecard, and Probabilistic Scorecard.

1. Model Training

This section contains the training procedures and results for signature kernel patch training on ERA5 reanalysis 64x32.

All model results across learning rate can be found directly in results/nets/calibration_metrics_SignatureKernel_{lr}.txt

To recacalculate Training and Evaluation, run the run.sh and runEvaluation.sh respectively, learning rate may be parsed.

Each model may take up to 48 hours to run, with most taking under 24 hours, and evaluation taking up to 2 hours on Warwick HPC clusters. 

2. Deterministic ScoreCard

Data is stored in FUXICard, IFSCard, and Newresults.

The Deterministic Scorecard can be created directly by running ScoreCardFinal.ipynb. 

To recalculate signature kernel results, run the additional two blocks in ScoreCardFinal.ipynb to generate Newresults.

To recalculate WeatherBench data calculations run WeatherBenchcalls.ipynb. 

To recalculate all data for this scorecard, it may take ~1.5-2 hours on an individual computer.

3. Probabilistic ScoreCard

Data is stored in WeatherData and Signature folders.

The Probabilistic Scorecard can be created directly by running ScoreCardFinal.ipynb

To recalculate signature data, run ProbScoreCard.py (optionally using runcompute.sh).

To recalculate WeatherBench data, run WeatherBenchCalls.ipynb.

This scorecard takes significantly longer to run recalculations (50x data for probabilistic), recommended to consider cluster usage for runcompute.sh.


