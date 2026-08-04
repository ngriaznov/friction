# Paired stock/antislop mining report

Stock/antislop paired mining corpus only (`corpus/paired/`, never manifest-tracked, never split). 55 stock doc(s), 55 antislop doc(s). eps=0.4, min_antislop_token_freq=8 (provisional — scaled down from mine-inventory's min_human_token_freq=25 for the much smaller paired corpus, pending recalibration once the real paired corpus exists), min_stock_count(n2/n3/n4)=8/5/4, top=50.
The `human_train_count` column is a read-only cross-check against `corpus/human` train-split docs only — it never touches `corpus/llm`, never contributes to scoring, and is this report's only bridge back to real human text.

## 2-gram
Total n-gram tokens: stock=30487, antislop=31308.

### stock-favored
| n-gram | stock count | antislop count | score | human_train_count |
|---|---|---|---|---|
| or a | 8 | 0 | 21.5655 | 37 |
| the email | 10 | 1 | 7.6286 | 0 |
| like any | 9 | 1 | 6.8951 | 3 |
| but the | 8 | 1 | 6.1616 | 45 |
| it's not | 8 | 1 | 6.1616 | 33 |
| move cursor | 12 | 2 | 5.3058 | 0 |
| crucial for | 15 | 3 | 4.6514 | 1 |
| all the | 9 | 2 | 4.0221 | 104 |
| feel free | 8 | 2 | 3.5943 | 8 |
| free to | 8 | 2 | 3.5943 | 12 |
| may be | 8 | 2 | 3.5943 | 71 |
| me know | 11 | 3 | 3.4432 | 2 |
| building a | 10 | 3 | 3.1412 | 4 |
| the initial | 12 | 4 | 2.8941 | 10 |
| we were | 12 | 4 | 2.8941 | 23 |
| know if | 11 | 4 | 2.6607 | 5 |
| post draft | 11 | 4 | 2.6607 | 0 |
| related to | 11 | 4 | 2.6607 | 14 |
| the main | 11 | 4 | 2.6607 | 44 |
| allow you | 8 | 3 | 2.5371 | 13 |
| as well | 8 | 3 | 2.5371 | 96 |
| can also | 8 | 3 | 2.5371 | 63 |
| we have | 8 | 3 | 2.5371 | 94 |
| allows you | 10 | 4 | 2.4273 | 23 |
| the data | 10 | 4 | 2.4273 | 42 |
| the root | 10 | 4 | 2.4273 | 12 |
| your specific | 19 | 8 | 2.3717 | 1 |
| built in | 12 | 5 | 2.3581 | 35 |
| to do | 9 | 4 | 2.1939 | 93 |
| you need | 9 | 4 | 2.1939 | 61 |
| free list | 13 | 6 | 2.1501 | 0 |
| they are | 13 | 6 | 2.1501 | 104 |
| into the | 16 | 8 | 2.0050 | 94 |
| is crucial | 16 | 8 | 2.0050 | 1 |
| the end | 14 | 7 | 1.9983 | 45 |
| the problem | 10 | 5 | 1.9778 | 28 |
| here's how | 8 | 4 | 1.9605 | 8 |
| the worker | 8 | 4 | 1.9605 | 3 |
| you to | 27 | 14 | 1.9540 | 72 |
| with the | 44 | 23 | 1.9485 | 320 |
| us to | 13 | 7 | 1.8596 | 37 |
| one of | 11 | 6 | 1.8292 | 115 |
| the same | 18 | 10 | 1.8169 | 251 |
| number of | 16 | 9 | 1.7917 | 109 |
| a full | 9 | 5 | 1.7876 | 13 |
| over the | 9 | 5 | 1.7876 | 51 |
| setting up | 9 | 5 | 1.7876 | 12 |
| the next | 9 | 5 | 1.7876 | 45 |
| front matter | 14 | 8 | 1.7605 | 2 |
| notes at | 12 | 7 | 1.7208 | 0 |

### antislop-favored
antislop-favored entries are diagnostic only — they show what the antislop model avoided, never a source of substitution-pair replacement text; any replacement text must be hand-picked from corpus/human train docs per inventory-en-v1.toml's curation convention.
| n-gram | stock count | antislop count | score | human_train_count |
|---|---|---|---|---|
| where the | 2 | 12 | 5.0312 | 38 |
| the app | 2 | 10 | 4.2197 | 11 |
| you must | 2 | 9 | 3.8140 | 14 |
| the overall | 3 | 12 | 3.5514 | 7 |
| you are | 3 | 12 | 3.5514 | 159 |
| for our | 5 | 19 | 3.4984 | 11 |
| a short | 2 | 8 | 3.4082 | 9 |
| the more | 2 | 8 | 3.4082 | 15 |
| this post | 2 | 8 | 3.4082 | 25 |
| used to | 2 | 8 | 3.4082 | 79 |
| is critical | 3 | 10 | 2.9786 | 0 |
| to create | 3 | 10 | 2.9786 | 78 |
| to our | 4 | 12 | 2.7443 | 25 |
| because it | 3 | 9 | 2.6922 | 26 |
| here's an | 3 | 9 | 2.6922 | 9 |
| and why | 3 | 8 | 2.4058 | 13 |
| is to | 3 | 8 | 2.4058 | 91 |
| it is | 3 | 8 | 2.4058 | 311 |
| of this | 6 | 15 | 2.3432 | 87 |
| the script | 4 | 10 | 2.3017 | 10 |
| the very | 4 | 10 | 2.3017 | 21 |
| this allows | 4 | 10 | 2.3017 | 8 |
| it's a | 7 | 17 | 2.2897 | 37 |
| in our | 8 | 18 | 2.1330 | 37 |
| a huge | 4 | 9 | 2.0803 | 14 |
| combination of | 4 | 9 | 2.0803 | 7 |
| than the | 4 | 9 | 2.0803 | 28 |
| to prevent | 4 | 9 | 2.0803 | 11 |
| your application | 5 | 11 | 2.0558 | 30 |
| from a | 6 | 13 | 2.0388 | 74 |
| what we | 6 | 12 | 1.8867 | 12 |
| the cursor | 5 | 10 | 1.8754 | 4 |
| adjust the | 4 | 8 | 1.8590 | 0 |
| an example | 4 | 8 | 1.8590 | 46 |
| ensure that | 4 | 8 | 1.8590 | 11 |
| there are | 4 | 8 | 1.8590 | 150 |
| this command | 4 | 8 | 1.8590 | 4 |
| the entire | 8 | 15 | 1.7853 | 22 |
| would be | 6 | 11 | 1.7345 | 71 |
| due to | 10 | 18 | 1.7228 | 47 |
| level of | 5 | 9 | 1.6951 | 18 |
| the current | 13 | 22 | 1.6278 | 62 |
| of your | 18 | 30 | 1.6088 | 50 |
| along with | 5 | 8 | 1.5148 | 30 |
| and how | 5 | 8 | 1.5148 | 41 |
| the biggest | 5 | 8 | 1.5148 | 1 |
| the primary | 5 | 8 | 1.5148 | 4 |
| to avoid | 5 | 8 | 1.5148 | 19 |
| use the | 5 | 8 | 1.5148 | 125 |
| dealing with | 11 | 17 | 1.4863 | 8 |

## 3-gram
Total n-gram tokens: stock=25696, antislop=26611.

### stock-favored
| n-gram | stock count | antislop count | score | human_train_count |
|---|---|---|---|---|
| need to be | 6 | 0 | 16.5697 | 16 |
| one of our | 5 | 0 | 13.9807 | 8 |
| to access the | 5 | 0 | 13.9807 | 1 |
| with the following | 5 | 0 | 13.9807 | 10 |
| is crucial for | 9 | 1 | 6.9534 | 0 |
| on your specific | 7 | 1 | 5.4739 | 0 |
| aimed for a | 6 | 1 | 4.7342 | 0 |
| i've included some | 6 | 1 | 4.7342 | 0 |
| so you can | 6 | 1 | 4.7342 | 19 |
| you for your | 5 | 1 | 3.9945 | 0 |
| feel free to | 8 | 2 | 3.6246 | 8 |
| this is crucial | 8 | 2 | 3.6246 | 0 |
| me know if | 11 | 3 | 3.4723 | 1 |
| the end of | 7 | 2 | 3.1931 | 16 |
| end of the | 6 | 2 | 2.7616 | 13 |
| path to your | 6 | 2 | 2.7616 | 0 |
| allow you to | 8 | 3 | 2.5586 | 11 |
| allows you to | 10 | 4 | 2.4478 | 23 |
| a draft email | 5 | 2 | 2.3301 | 0 |
| a list of | 5 | 2 | 2.3301 | 23 |
| is designed to | 5 | 2 | 2.3301 | 8 |
| beginning of the | 7 | 3 | 2.2540 | 1 |
| blog post draft | 9 | 4 | 2.2124 | 0 |
| at the end | 10 | 5 | 1.9945 | 14 |
| you want to | 10 | 5 | 1.9945 | 85 |
| the beginning of | 7 | 4 | 1.7417 | 5 |
| you have a | 7 | 4 | 1.7417 | 28 |
| notes at the | 12 | 7 | 1.7353 | 0 |
| this is where | 12 | 7 | 1.7353 | 6 |
| is a common | 5 | 3 | 1.6448 | 3 |
| thank you for | 5 | 3 | 1.6448 | 3 |
| wanted to share | 5 | 3 | 1.6448 | 1 |
| your photo library | 5 | 3 | 1.6448 | 0 |
| designed to be | 6 | 4 | 1.5063 | 7 |
| multi stage builds | 6 | 4 | 1.5063 | 0 |
| if you have | 10 | 7 | 1.4555 | 64 |
| this is a | 14 | 10 | 1.4339 | 54 |
| read the notes | 7 | 5 | 1.4192 | 0 |
| the notes at | 7 | 5 | 1.4192 | 0 |
| we want to | 7 | 5 | 1.4192 | 26 |
| when dealing with | 7 | 5 | 1.4192 | 1 |
| the number of | 8 | 6 | 1.3592 | 22 |
| food bank name | 9 | 7 | 1.3155 | 0 |
| i've aimed for | 15 | 12 | 1.2862 | 0 |
| to your specific | 5 | 4 | 1.2710 | 0 |
| here's a draft | 9 | 8 | 1.1589 | 0 |
| here's a blog | 9 | 9 | 1.0356 | 0 |
| a blog post | 9 | 10 | 0.9360 | 1 |
| based on your | 9 | 10 | 0.9360 | 1 |
| this is the | 8 | 9 | 0.9254 | 34 |

### antislop-favored
antislop-favored entries are diagnostic only — they show what the antislop model avoided, never a source of substitution-pair replacement text; any replacement text must be hand-picked from corpus/human train docs per inventory-en-v1.toml's curation convention.
| n-gram | stock count | antislop count | score | human_train_count |
|---|---|---|---|---|
| to create a | 0 | 6 | 15.4499 | 33 |
| the current line | 0 | 5 | 13.0358 | 1 |
| here's a breakdown | 1 | 5 | 3.7245 | 0 |
| the order of | 1 | 5 | 3.7245 | 6 |
| let's break down | 2 | 6 | 2.5750 | 0 |
| this is critical | 2 | 6 | 2.5750 | 0 |
| a combination of | 3 | 7 | 2.1016 | 4 |
| at the very | 4 | 7 | 1.6240 | 4 |
| parental leave policy | 4 | 7 | 1.6240 | 0 |
| for a while | 3 | 5 | 1.5336 | 9 |
| the lifetime of | 3 | 5 | 1.5336 | 0 |
| a covering index | 4 | 6 | 1.4045 | 0 |
| you can use | 5 | 6 | 1.1444 | 47 |
| please read the | 7 | 8 | 1.0961 | 2 |
| this is the | 8 | 9 | 1.0806 | 34 |
| a blog post | 9 | 10 | 1.0683 | 1 |
| based on your | 9 | 10 | 1.0683 | 1 |
| here's a blog | 9 | 9 | 0.9656 | 0 |
| here's a draft | 9 | 8 | 0.8629 | 0 |
| i've aimed for | 15 | 12 | 0.7775 | 0 |
| food bank name | 9 | 7 | 0.7602 | 0 |
| the number of | 8 | 6 | 0.7357 | 22 |
| read the notes | 7 | 5 | 0.7046 | 0 |
| the notes at | 7 | 5 | 0.7046 | 0 |
| we want to | 7 | 5 | 0.7046 | 26 |
| when dealing with | 7 | 5 | 0.7046 | 1 |
| this is a | 14 | 10 | 0.6974 | 54 |
| if you have | 10 | 7 | 0.6871 | 64 |
| notes at the | 12 | 7 | 0.5763 | 0 |
| this is where | 12 | 7 | 0.5763 | 6 |
| at the end | 10 | 5 | 0.5014 | 14 |
| you want to | 10 | 5 | 0.5014 | 85 |

## 4-gram
Total n-gram tokens: stock=21539, antislop=22589.

### stock-favored
| n-gram | stock count | antislop count | score | human_train_count |
|---|---|---|---|---|
| based on your specific | 5 | 0 | 14.1581 | 0 |
| move cursor to the | 4 | 0 | 11.5362 | 0 |
| please read those notes | 4 | 0 | 11.5362 | 0 |
| the path to your | 4 | 0 | 11.5362 | 0 |
| to the end of | 4 | 0 | 11.5362 | 1 |
| with the following content | 4 | 0 | 11.5362 | 0 |
| i've aimed for a | 5 | 1 | 4.0452 | 0 |
| if you have a | 5 | 1 | 4.0452 | 16 |
| thank you for your | 5 | 1 | 4.0452 | 0 |
| the end of the | 5 | 1 | 4.0452 | 10 |
| the beginning of the | 7 | 2 | 3.2336 | 1 |
| here's a draft email | 5 | 2 | 2.3597 | 0 |
| a blog post draft | 9 | 4 | 2.2405 | 0 |
| notes at the end | 4 | 2 | 1.9227 | 0 |
| please read the notes | 7 | 5 | 1.4372 | 0 |
| read the notes at | 7 | 5 | 1.4372 | 0 |
| the notes at the | 7 | 5 | 1.4372 | 0 |
| at the very end | 4 | 3 | 1.3572 | 0 |
| i've aimed for clarity | 4 | 3 | 1.3572 | 0 |
| here's a blog post | 9 | 9 | 1.0487 | 0 |
| notes at the very | 4 | 5 | 0.8545 | 0 |

### antislop-favored
antislop-favored entries are diagnostic only — they show what the antislop model avoided, never a source of substitution-pair replacement text; any replacement text must be hand-picked from corpus/human train docs per inventory-en-v1.toml's curation convention.
| n-gram | stock count | antislop count | score | human_train_count |
|---|---|---|---|---|
| model or description e | 0 | 4 | 10.4887 | 0 |
| at food bank name | 1 | 4 | 2.9968 | 0 |
| here's a breakdown of | 1 | 4 | 2.9968 | 0 |
| if you have any | 1 | 4 | 2.9968 | 6 |
| notes at the very | 4 | 5 | 1.1702 | 0 |
| here's a blog post | 9 | 9 | 0.9535 | 0 |
| please read the notes | 7 | 5 | 0.6958 | 0 |
| read the notes at | 7 | 5 | 0.6958 | 0 |
| the notes at the | 7 | 5 | 0.6958 | 0 |
| a blog post draft | 9 | 4 | 0.4463 | 0 |
