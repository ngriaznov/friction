# Separation report

Dev split, human vs llm. AUC is the Mann-Whitney U statistic, tie-corrected via midranks, oriented so AUC > 0.5 always means the metric separates the two classes (see `direction` for which one scores higher). Combined score: the mean, over a document's genre's *included* metrics (envelope pack `include` flag, a train-internal AUC >= 0.5500 rule decided entirely by `corpus-tool envelope` — see each genre's "Combined-score metrics" line below for which metrics that excluded and why), of a per-metric normalized directional exceedance beyond that metric's train-human envelope band (0.0 inside the band, else the distance to the nearer edge over the band width, capped at 1.0); its own AUC uses the same Mann-Whitney method. All figures to 4 decimal places.

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
| llm_favored_phrase_rate | 13 | 10 | 0.9846 | llm higher |
| human_favored_phrase_rate | 13 | 10 | 1.0000 | llm lower |
| heading_density | 13 | 10 | 0.5038 | llm higher |
| list_item_density | 13 | 10 | 0.8000 | llm higher |
| bold_span_density | 13 | 10 | 0.9538 | llm higher |
| sentence_opener_repeat_rate | 13 | 10 | 0.8462 | llm lower |
| top_opener_concentration | 13 | 10 | 0.5038 | llm higher |

Summary: docs — human n=13, llm n=10, combined-score AUC = 0.9154 (llm higher)
Combined-score metrics: 15 of 21 included; excluded: bullet_parallelism (train AUC 0.5051, llm higher), em_dash_density (train AUC 0.5202, llm lower), heading_density (train AUC 0.5463, llm lower), not_just_but_rate (train AUC 0.5024, llm higher), paragraph_shape_mean (train AUC 0.5191, llm lower), ritual_marker_rate (train AUC 0.5000, llm higher).

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
| llm_favored_phrase_rate | 15 | 10 | 1.0000 | llm higher |
| human_favored_phrase_rate | 15 | 10 | 0.8800 | llm lower |
| heading_density | 15 | 10 | 0.6000 | llm lower |
| list_item_density | 15 | 10 | 0.5400 | llm higher |
| bold_span_density | 15 | 10 | 0.9300 | llm higher |
| sentence_opener_repeat_rate | 15 | 10 | 0.6633 | llm lower |
| top_opener_concentration | 15 | 10 | 0.7667 | llm higher |

Summary: blog — human n=15, llm n=10, combined-score AUC = 0.9933 (llm higher)
Combined-score metrics: 19 of 21 included; excluded: paragraph_shape_mean (train AUC 0.5147, llm lower), sentence_length_mean (train AUC 0.5166, llm higher).

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
| llm_favored_phrase_rate | 12 | 10 | 0.8750 | llm higher |
| human_favored_phrase_rate | 12 | 10 | 0.9167 | llm lower |
| heading_density | 12 | 10 | 0.5750 | llm lower |
| list_item_density | 12 | 10 | 0.8333 | llm lower |
| bold_span_density | 12 | 10 | 0.8500 | llm higher |
| sentence_opener_repeat_rate | 12 | 10 | 0.5333 | llm lower |
| top_opener_concentration | 12 | 10 | 0.8167 | llm higher |

Summary: readme — human n=12, llm n=10, combined-score AUC = 0.9500 (llm higher)
Combined-score metrics: 14 of 21 included; excluded: bullet_parallelism (train AUC 0.5336, llm lower), contraction_ratio (train AUC 0.5147, llm higher), discourse_marker_density (train AUC 0.5252, llm higher), em_dash_density (train AUC 0.5130, llm lower), not_just_but_rate (train AUC 0.5000, llm higher), ritual_marker_rate (train AUC 0.5109, llm higher), semicolon_density (train AUC 0.5454, llm lower).

## email

| metric | human n | llm n | AUC | direction |
|---|---|---|---|---|
| sentence_length_mean | 7 | 10 | 1.0000 | llm higher |
| sentence_length_stddev | 7 | 10 | 0.7714 | llm higher |
| sentence_length_cv | 7 | 10 | 1.0000 | llm lower |
| discourse_marker_density | 7 | 10 | 0.6500 | llm higher |
| triad_rate | 7 | 10 | 0.7500 | llm higher |
| contraction_ratio | 7 | 10 | 0.7429 | llm lower |
| bullet_parallelism | 7 | 10 | 0.5143 | llm higher |
| paragraph_shape_mean | 7 | 10 | 0.9000 | llm higher |
| paragraph_shape_cv | 7 | 10 | 0.7714 | llm higher |
| em_dash_density | 7 | 10 | 0.6714 | llm lower |
| semicolon_density | 7 | 10 | 0.5714 | llm lower |
| participial_closer_rate | 7 | 10 | 0.7714 | llm higher |
| not_just_but_rate | 7 | 10 | 0.6000 | llm higher |
| ritual_marker_rate | 7 | 10 | 0.5000 | llm higher |
| llm_favored_phrase_rate | 7 | 10 | 1.0000 | llm higher |
| human_favored_phrase_rate | 7 | 10 | 0.7571 | llm lower |
| heading_density | 7 | 10 | 0.8857 | llm lower |
| list_item_density | 7 | 10 | 0.8286 | llm lower |
| bold_span_density | 7 | 10 | 0.7857 | llm higher |
| sentence_opener_repeat_rate | 7 | 10 | 0.6143 | llm higher |
| top_opener_concentration | 7 | 10 | 0.5857 | llm higher |

Summary: email — human n=7, llm n=10, combined-score AUC = 1.0000 (llm higher)
Combined-score metrics: 18 of 21 included; excluded: contraction_ratio (train AUC 0.5072, llm lower), not_just_but_rate (train AUC 0.5036, llm lower), semicolon_density (train AUC 0.5355, llm lower).

## forum

| metric | human n | llm n | AUC | direction |
|---|---|---|---|---|
| sentence_length_mean | 11 | 10 | 0.6909 | llm lower |
| sentence_length_stddev | 11 | 10 | 0.7000 | llm lower |
| sentence_length_cv | 11 | 10 | 0.5909 | llm lower |
| discourse_marker_density | 11 | 10 | 0.6364 | llm higher |
| triad_rate | 11 | 10 | 0.5182 | llm higher |
| contraction_ratio | 11 | 10 | 0.5318 | llm lower |
| bullet_parallelism | 11 | 10 | 0.6591 | llm higher |
| paragraph_shape_mean | 11 | 10 | 0.7273 | llm lower |
| paragraph_shape_cv | 11 | 10 | 0.6636 | llm lower |
| em_dash_density | 11 | 10 | 0.5000 | llm higher |
| semicolon_density | 11 | 10 | 0.6000 | llm higher |
| participial_closer_rate | 11 | 10 | 0.6273 | llm higher |
| not_just_but_rate | 11 | 10 | 0.5500 | llm higher |
| ritual_marker_rate | 11 | 10 | 0.6000 | llm higher |
| llm_favored_phrase_rate | 11 | 10 | 0.7636 | llm higher |
| human_favored_phrase_rate | 11 | 10 | 0.9182 | llm lower |
| heading_density | 11 | 10 | 0.5318 | llm higher |
| list_item_density | 11 | 10 | 0.7182 | llm higher |
| bold_span_density | 11 | 10 | 0.7455 | llm higher |
| sentence_opener_repeat_rate | 11 | 10 | 0.8409 | llm lower |
| top_opener_concentration | 11 | 10 | 0.5045 | llm lower |

Summary: forum — human n=11, llm n=10, combined-score AUC = 0.8636 (llm higher)
Combined-score metrics: 17 of 21 included; excluded: bullet_parallelism (train AUC 0.5134, llm higher), em_dash_density (train AUC 0.5047, llm lower), top_opener_concentration (train AUC 0.5126, llm higher), triad_rate (train AUC 0.5104, llm higher).

## Combined-score gate

Genres whose combined-score AUC reaches 0.8500: 5 of 5 (target: at least 3). Status: MET.

| genre | combined-score AUC | reaches 0.8500 |
|---|---|---|
| docs | 0.9154 | yes |
| blog | 0.9933 | yes |
| readme | 0.9500 | yes |
| email | 1.0000 | yes |
| forum | 0.8636 | yes |
