# UWS label models on the jev-ws fast-decision pools

Primary metric ×100, mean over tasks, cross-fitted (2-fold) protocol of `run_tuned_aggregation.py`. Wins = tasks where the method beats the column method.

## Pool `original` (37 tasks; best fixed model: `kev-9b`)

| method | mean | vs prob_avg | wins vs prob_avg | vs snorkel_tuned | wins vs snorkel_tuned | vs best_single_cv | vs kev-9b |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_single_cv | 77.04 | +2.49 | 30 | +3.15 | 24 | +0.00 | +2.85 |
| majority | 73.00 | -1.55 | 11 | -0.90 | 13 | -4.04 | -1.19 |
| prob_avg | 74.55 | +0.00 | 0 | +0.66 | 13 | -2.49 | +0.36 |
| snorkel_tuned | 73.89 | -0.66 | 24 | +0.00 | 0 | -3.15 | -0.29 |
| kev-9b | 74.19 | -0.36 | 22 | +0.29 | 13 | -2.85 | +0.00 |
| uws_gauss | 75.37 | +0.82 | 29 | +1.48 | 17 | -1.67 | +1.19 |
| uws_gauss_fam | 73.01 | -1.54 | 15 | -0.88 | 10 | -4.03 | -1.17 |
| uws_linpool | 75.48 | +0.93 | 27 | +1.59 | 26 | -1.56 | +1.29 |
| uws_linpool_fam | 72.35 | -2.20 | 17 | -1.54 | 15 | -4.69 | -1.84 |
| uws_logpool | 74.14 | -0.41 | 27 | +0.25 | 19 | -2.90 | -0.05 |
| uws_logpool_fam | 70.92 | -3.63 | 9 | -2.97 | 11 | -6.12 | -3.26 |
| uws_ordinal | 73.72 | -0.83 | 25 | -0.17 | 19 | -3.32 | -0.46 |
| uws_ordinal_fam | 70.50 | -4.05 | 9 | -3.39 | 11 | -6.54 | -3.69 |
| uws_tuned | 75.59 | +1.04 | 32 | +1.70 | 23 | -1.45 | +1.41 |
| uws_vote | 72.21 | -2.34 | 24 | -1.68 | 15 | -4.83 | -1.97 |
| uws_vote_fam | 68.17 | -6.38 | 10 | -5.72 | 6 | -8.87 | -6.02 |

By category (mean primary ×100):

| category | prob_avg | snorkel_tuned | uws_gauss | uws_logpool | uws_tuned |
|---|---|---|---|---|---|
| Classification | 83.05 | 83.18 | 82.87 | 83.63 | 84.11 |
| Grounding | 72.33 | 76.70 | 72.99 | 73.59 | 73.99 |
| Judge | 56.54 | 51.29 | 57.92 | 58.60 | 56.71 |
| Knowledge/MC | 77.13 | 78.61 | 79.25 | 79.25 | 79.40 |
| Legal | 40.61 | 32.02 | 43.96 | 25.22 | 42.90 |
| Routing | 84.22 | 85.28 | 83.17 | 83.17 | 85.78 |
| Safety | 78.50 | 73.13 | 78.88 | 73.47 | 76.08 |
| Semantic | 75.25 | 77.53 | 75.71 | 76.12 | 76.81 |

Per task, Judge/Safety/Legal (where Snorkel struggled):

| task | best_single_cv | prob_avg | snorkel_tuned | uws_tuned | uws_gauss | uws_gauss_fam | uws_ordinal | uws_logpool |
|---|---|---|---|---|---|---|---|---|
| agb_de | 18.64 | 10.39 | 2.56 | 12.23 | 14.35 | 14.35 | 12.50 | 12.50 |
| helpsteer2 | 41.41 | 40.18 | 33.61 | 31.16 | 40.73 | 40.73 | 37.35 | 40.79 |
| openai_moderation | 59.68 | 64.60 | 54.35 | 53.59 | 64.96 | 64.96 | 56.08 | 56.08 |
| prompt_injections | 94.83 | 92.24 | 79.31 | 93.10 | 94.83 | 94.83 | 82.76 | 82.76 |
| sst5 | 57.92 | 54.84 | 54.43 | 56.52 | 51.81 | 51.81 | 56.24 | 55.43 |
| stsb | 87.42 | 85.60 | 83.74 | 87.59 | 87.59 | 87.59 | 85.79 | 86.60 |
| summeval | 49.56 | 45.54 | 33.36 | 51.57 | 51.57 | 51.57 | 39.58 | 51.58 |
| toxic_chat | 76.61 | 78.13 | 78.44 | 78.13 | 76.58 | 76.58 | 77.72 | 77.72 |
| toxigen | 79.15 | 79.04 | 80.43 | 79.47 | 79.15 | 79.15 | 77.34 | 77.34 |
| unfair_tos | 72.73 | 70.83 | 61.48 | 73.58 | 73.58 | 73.58 | 37.94 | 37.94 |

`uws_tuned` selections over folds: uws_linpool 26, uws_logpool 15, uws_vote 15, uws_gauss 10, uws_linpool_fam 4, uws_vote_fam 3, uws_logpool_fam 1

## Pool `strong` (37 tasks; best fixed model: `decider-gemma4-31b`)

| method | mean | vs prob_avg | wins vs prob_avg | vs snorkel_tuned | wins vs snorkel_tuned | vs best_single_cv | vs decider-gemma4-31b |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_single_cv | 80.05 | +0.46 | 19 | +1.27 | 24 | +0.00 | +1.69 |
| majority | 78.79 | -0.80 | 9 | +0.01 | 17 | -1.26 | +0.43 |
| prob_avg | 79.59 | +0.00 | 0 | +0.81 | 25 | -0.46 | +1.23 |
| snorkel_tuned | 78.78 | -0.81 | 11 | +0.00 | 0 | -1.27 | +0.42 |
| decider-gemma4-31b | 78.36 | -1.23 | 13 | -0.42 | 16 | -1.69 | +0.00 |
| uws_gauss | 79.32 | -0.28 | 9 | +0.54 | 22 | -0.73 | +0.96 |
| uws_gauss_fam | 79.31 | -0.29 | 14 | +0.53 | 20 | -0.74 | +0.95 |
| uws_linpool | 79.53 | -0.06 | 18 | +0.75 | 26 | -0.52 | +1.17 |
| uws_linpool_fam | 79.62 | +0.02 | 15 | +0.83 | 25 | -0.43 | +1.25 |
| uws_logpool | 79.28 | -0.31 | 7 | +0.50 | 23 | -0.77 | +0.92 |
| uws_logpool_fam | 79.40 | -0.19 | 17 | +0.62 | 21 | -0.65 | +1.04 |
| uws_ordinal | 79.09 | -0.50 | 5 | +0.31 | 23 | -0.96 | +0.73 |
| uws_ordinal_fam | 79.25 | -0.34 | 14 | +0.47 | 21 | -0.80 | +0.89 |
| uws_tuned | 79.76 | +0.17 | 21 | +0.98 | 24 | -0.29 | +1.40 |
| uws_vote | 77.25 | -2.35 | 8 | -1.54 | 15 | -2.80 | -1.11 |
| uws_vote_fam | 77.31 | -2.28 | 9 | -1.47 | 13 | -2.74 | -1.05 |

By category (mean primary ×100):

| category | prob_avg | snorkel_tuned | uws_gauss | uws_logpool | uws_tuned |
|---|---|---|---|---|---|
| Classification | 82.72 | 83.72 | 82.39 | 82.43 | 82.57 |
| Grounding | 76.88 | 78.35 | 77.08 | 76.75 | 78.28 |
| Judge | 62.18 | 59.48 | 61.14 | 62.52 | 62.60 |
| Knowledge/MC | 87.57 | 87.28 | 87.57 | 87.57 | 87.88 |
| Legal | 49.24 | 40.48 | 50.40 | 45.56 | 48.61 |
| Routing | 89.44 | 89.08 | 88.86 | 88.86 | 89.33 |
| Safety | 84.22 | 78.92 | 83.35 | 84.23 | 84.31 |
| Semantic | 80.12 | 82.04 | 79.95 | 79.87 | 80.47 |

Per task, Judge/Safety/Legal (where Snorkel struggled):

| task | best_single_cv | prob_avg | snorkel_tuned | uws_tuned | uws_gauss | uws_gauss_fam | uws_ordinal | uws_logpool |
|---|---|---|---|---|---|---|---|---|
| agb_de | 32.10 | 22.78 | 22.54 | 22.22 | 27.85 | 27.85 | 19.18 | 19.18 |
| helpsteer2 | 42.43 | 44.34 | 41.09 | 44.42 | 44.03 | 44.03 | 42.27 | 44.28 |
| openai_moderation | 68.18 | 72.56 | 56.02 | 72.53 | 72.91 | 72.91 | 72.48 | 72.48 |
| prompt_injections | 95.69 | 94.83 | 90.52 | 94.83 | 94.83 | 94.83 | 95.69 | 95.69 |
| sst5 | 57.87 | 58.10 | 57.01 | 58.33 | 55.34 | 55.34 | 57.47 | 58.33 |
| stsb | 90.15 | 90.75 | 89.14 | 90.89 | 90.79 | 90.79 | 89.48 | 90.72 |
| summeval | 54.08 | 55.52 | 50.68 | 56.77 | 54.41 | 54.41 | 53.85 | 56.77 |
| toxic_chat | 80.00 | 80.44 | 80.11 | 80.93 | 79.37 | 79.37 | 80.44 | 80.44 |
| toxigen | 88.72 | 89.04 | 89.04 | 88.94 | 86.28 | 86.28 | 88.30 | 88.30 |
| unfair_tos | 70.02 | 75.69 | 58.42 | 75.00 | 72.95 | 72.95 | 71.94 | 71.94 |

`uws_tuned` selections over folds: uws_linpool 18, uws_linpool_fam 17, uws_logpool 15, uws_logpool_fam 10, uws_vote_fam 7, uws_gauss 4, uws_vote 3

