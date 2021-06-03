# Universalizing Weak Supervision



### System Requirements

* Anaconda
* Python 3.6
* Pytorch
* See environment.yml for details

### Environment Setup

We recommend you create a conda environment as follows

```
conda env create -f environment.yml
```

and activate it with

```
conda activate ws-cardinality
```

### Running Experiments

* Full ranking, partial ranking experiments
  * notebooks/{board-games, imdb-tmdb}/RankingExperiments.ipynb
    * To play with configurations, you can look into configs {board-games, imdb-tmdb}_ranking_experiment.yaml
    * Mainly changed configurations are
      * n_train
      * n_test
      * p: null # 0.2 | 0.4 | 0.6 | 0.8 (observational probability)
      * num_LFs: 3 # 6 | 9 | 12
      * inference_rule: weighted kemeny # | snorkel | kemeny | pairwise_majority | weighted_pairwise_majority
        * Note that snorkel is our baseline. kemeny and pariwise_majority is a basic ranking aggregation method for full rankings, and partial rankings respectively.
* Regression experiments
  * notebooks/{board-games, imdb-tmdb}/RegressionExperiments.ipynb

* Synthetic data experiments
  * notebooks/synthetic