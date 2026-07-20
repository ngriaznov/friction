# Separation report

Dev split, human vs llm. AUC is the Mann-Whitney U statistic, tie-corrected via midranks, oriented so AUC > 0.5 always means the metric separates the two classes (see `direction` for which one scores higher). Combined score: the fraction of a document's 14 metrics falling outside its genre's train-human envelope; its own AUC uses the same method. All figures to 4 decimal places.

## docs

| metric | human n | llm n | AUC | direction |
|---|---|---|---|---|
| sentence_length_mean | 13 | 10 | 0.6615 | llm higher |
| sentence_length_stddev | 13 | 10 | 0.5538 | llm lower |
| sentence_length_cv | 13 | 10 | 0.8538 | llm lower |
| discourse_marker_density | 13 | 10 | 0.6769 | llm lower |
| triad_rate | 13 | 10 | 0.7538 | llm higher |
| contraction_ratio | 13 | 10 | 0.5385 | llm lower |
| bullet_parallelism | 13 | 10 | 0.5538 | llm higher |
| paragraph_shape_mean | 13 | 10 | 0.5000 | llm higher |
| paragraph_shape_cv | 13 | 10 | 0.7077 | llm lower |
| em_dash_density | 13 | 10 | 0.5769 | llm lower |
| semicolon_density | 13 | 10 | 0.5500 | llm lower |
| participial_closer_rate | 13 | 10 | 0.8154 | llm higher |
| not_just_but_rate | 13 | 10 | 0.6231 | llm higher |
| ritual_marker_rate | 13 | 10 | 0.5000 | llm higher |

Summary: docs — human n=13, llm n=10, combined-score AUC = 0.7923 (llm higher)

## blog

| metric | human n | llm n | AUC | direction |
|---|---|---|---|---|
| sentence_length_mean | 15 | 10 | 0.7000 | llm higher |
| sentence_length_stddev | 15 | 10 | 0.6867 | llm lower |
| sentence_length_cv | 15 | 10 | 0.8400 | llm lower |
| discourse_marker_density | 15 | 10 | 0.7867 | llm higher |
| triad_rate | 15 | 10 | 0.8000 | llm higher |
| contraction_ratio | 15 | 10 | 0.8100 | llm higher |
| bullet_parallelism | 15 | 10 | 0.5767 | llm higher |
| paragraph_shape_mean | 15 | 10 | 0.6800 | llm higher |
| paragraph_shape_cv | 15 | 10 | 0.6800 | llm lower |
| em_dash_density | 15 | 10 | 0.6433 | llm lower |
| semicolon_density | 15 | 10 | 0.6133 | llm lower |
| participial_closer_rate | 15 | 10 | 0.6500 | llm higher |
| not_just_but_rate | 15 | 10 | 0.6000 | llm higher |
| ritual_marker_rate | 15 | 10 | 0.5767 | llm higher |

Summary: blog — human n=15, llm n=10, combined-score AUC = 0.8067 (llm higher)

## readme

| metric | human n | llm n | AUC | direction |
|---|---|---|---|---|
| sentence_length_mean | 12 | 10 | 0.9333 | llm higher |
| sentence_length_stddev | 12 | 10 | 0.9083 | llm higher |
| sentence_length_cv | 12 | 10 | 0.8000 | llm lower |
| discourse_marker_density | 12 | 10 | 0.6083 | llm lower |
| triad_rate | 12 | 10 | 0.6083 | llm higher |
| contraction_ratio | 12 | 10 | 0.6667 | llm lower |
| bullet_parallelism | 12 | 10 | 0.6833 | llm lower |
| paragraph_shape_mean | 12 | 10 | 0.8542 | llm higher |
| paragraph_shape_cv | 12 | 10 | 0.8583 | llm higher |
| em_dash_density | 12 | 10 | 0.5833 | llm lower |
| semicolon_density | 12 | 10 | 0.6250 | llm lower |
| participial_closer_rate | 12 | 10 | 0.7667 | llm higher |
| not_just_but_rate | 12 | 10 | 0.5000 | llm higher |
| ritual_marker_rate | 12 | 10 | 0.5000 | llm higher |

Summary: readme — human n=12, llm n=10, combined-score AUC = 0.9583 (llm higher)

## email

| metric | human n | llm n | AUC | direction |
|---|---|---|---|---|
| sentence_length_mean | 7 | 10 | 1.0000 | llm higher |
| sentence_length_stddev | 7 | 10 | 0.7714 | llm higher |
| sentence_length_cv | 7 | 10 | 1.0000 | llm lower |
| discourse_marker_density | 7 | 10 | 0.6500 | llm higher |
| triad_rate | 7 | 10 | 0.7500 | llm higher |
| contraction_ratio | 7 | 10 | 0.7429 | llm lower |
| bullet_parallelism | 7 | 10 | 0.5000 | llm higher |
| paragraph_shape_mean | 7 | 10 | 0.9000 | llm higher |
| paragraph_shape_cv | 7 | 10 | 0.7714 | llm higher |
| em_dash_density | 7 | 10 | 0.6714 | llm lower |
| semicolon_density | 7 | 10 | 0.5714 | llm lower |
| participial_closer_rate | 7 | 10 | 0.7714 | llm higher |
| not_just_but_rate | 7 | 10 | 0.6000 | llm higher |
| ritual_marker_rate | 7 | 10 | 0.5000 | llm higher |

Summary: email — human n=7, llm n=10, combined-score AUC = 0.8786 (llm higher)

## forum

| metric | human n | llm n | AUC | direction |
|---|---|---|---|---|
| sentence_length_mean | 11 | 10 | 0.6909 | llm lower |
| sentence_length_stddev | 11 | 10 | 0.7000 | llm lower |
| sentence_length_cv | 11 | 10 | 0.5636 | llm lower |
| discourse_marker_density | 11 | 10 | 0.6091 | llm higher |
| triad_rate | 11 | 10 | 0.5182 | llm higher |
| contraction_ratio | 11 | 10 | 0.8500 | llm higher |
| bullet_parallelism | 11 | 10 | 0.6591 | llm higher |
| paragraph_shape_mean | 11 | 10 | 0.7273 | llm lower |
| paragraph_shape_cv | 11 | 10 | 0.6545 | llm lower |
| em_dash_density | 11 | 10 | 0.5000 | llm higher |
| semicolon_density | 11 | 10 | 0.9409 | llm lower |
| participial_closer_rate | 11 | 10 | 0.6273 | llm higher |
| not_just_but_rate | 11 | 10 | 0.5500 | llm higher |
| ritual_marker_rate | 11 | 10 | 0.6000 | llm higher |

Summary: forum — human n=11, llm n=10, combined-score AUC = 0.7773 (llm higher)

## Combined-score gate

Genres whose combined-score AUC reaches 0.8500: 2 of 5 (target: at least 3). Status: NOT MET.

| genre | combined-score AUC | reaches 0.8500 |
|---|---|---|
| docs | 0.7923 | no |
| blog | 0.8067 | no |
| readme | 0.9583 | yes |
| email | 0.8786 | yes |
| forum | 0.7773 | no |
