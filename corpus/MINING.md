# N-gram mining report

Train split only, pooled across genres. 629 document(s) (264 human, 365 llm). Scored via the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008, eq. 16; prior drawn from the two classes' own combined counts — see this module's doc comment for the exact formula). z > 0 is llm-favored, z < 0 is human-favored. Entries below --min-count (5) are omitted; top 120 per direction per order.

## 1-gram

Total n-gram tokens: llm=191905, human=231214. Scored vocabulary (>= min-count): 6789.

### llm-favored

| n-gram | llm count | human count | z | delta |
|---|---|---|---|---|
| your | 1547 | 874 | 10.5746 | 0.2492 |
| our | 944 | 508 | 8.6589 | 0.2637 |
| mysql | 132 | 10 | 6.4410 | 0.6514 |
| job | 170 | 30 | 6.4028 | 0.5376 |
| three | 248 | 80 | 6.2915 | 0.4071 |
| step | 195 | 47 | 6.2745 | 0.4757 |
| i'd | 125 | 10 | 6.2337 | 0.6461 |
| six | 113 | 6 | 6.1339 | 0.6806 |
| backup | 121 | 11 | 6.0475 | 0.6327 |
| perceptual | 95 | 0 | 6.0106 | 0.7553 |
| ensure | 179 | 45 | 5.9241 | 0.4664 |
| review | 131 | 18 | 5.9214 | 0.5790 |
| i | 1007 | 745 | 5.8798 | 0.1624 |
| lifetime | 95 | 3 | 5.7787 | 0.7097 |
| a | 5601 | 5654 | 5.5515 | 0.0604 |
| chirpline | 80 | 0 | 5.5155 | 0.7552 |
| roughly | 102 | 10 | 5.5011 | 0.6241 |
| rather | 176 | 53 | 5.4690 | 0.4242 |
| route | 85 | 3 | 5.4405 | 0.7045 |
| duplicates | 80 | 1 | 5.4307 | 0.7368 |
| weeks | 137 | 30 | 5.4226 | 0.4960 |
| migration | 130 | 26 | 5.4221 | 0.5141 |
| two | 410 | 233 | 5.3873 | 0.2464 |
| four | 118 | 20 | 5.3841 | 0.5447 |
| file | 518 | 326 | 5.3770 | 0.2143 |
| date | 157 | 46 | 5.2267 | 0.4308 |
| than | 376 | 212 | 5.2018 | 0.2488 |
| crucial | 80 | 4 | 5.1810 | 0.6846 |
| commit | 119 | 24 | 5.1755 | 0.5125 |
| me | 215 | 88 | 5.1269 | 0.3435 |
| database | 220 | 93 | 5.0760 | 0.3344 |
| every | 259 | 123 | 5.0537 | 0.3007 |
| hashing | 85 | 8 | 5.0472 | 0.6288 |
| index | 149 | 45 | 5.0252 | 0.4234 |
| per | 156 | 50 | 5.0038 | 0.4086 |
| conclusion | 93 | 13 | 4.9730 | 0.5763 |
| process | 257 | 125 | 4.9367 | 0.2936 |
| robust | 86 | 10 | 4.9324 | 0.6027 |
| backups | 79 | 7 | 4.9005 | 0.6354 |
| cost | 100 | 19 | 4.8205 | 0.5239 |
| bump | 61 | 0 | 4.8160 | 0.7552 |
| gerrit | 61 | 0 | 4.8160 | 0.7552 |
| bed | 99 | 19 | 4.7838 | 0.5220 |
| before | 312 | 176 | 4.7349 | 0.2486 |
| ensuring | 74 | 7 | 4.7062 | 0.6282 |
| significant | 105 | 24 | 4.6840 | 0.4870 |
| minutes | 103 | 23 | 4.6733 | 0.4919 |
| subject | 83 | 12 | 4.6681 | 0.5710 |
| adjust | 62 | 2 | 4.6640 | 0.7087 |
| table | 161 | 61 | 4.6530 | 0.3647 |
| raidz | 56 | 0 | 4.6143 | 0.7552 |
| cron | 58 | 1 | 4.5966 | 0.7299 |
| memory | 227 | 112 | 4.5836 | 0.2893 |
| nobody | 60 | 2 | 4.5820 | 0.7072 |
| costs | 64 | 4 | 4.5608 | 0.6682 |
| queue | 74 | 9 | 4.5433 | 0.5965 |
| nothing | 101 | 24 | 4.5361 | 0.4787 |
| lumen | 54 | 0 | 4.5312 | 0.7552 |
| transceiver | 54 | 0 | 4.5312 | 0.7552 |
| batch | 76 | 11 | 4.4659 | 0.5709 |
| nozzle | 52 | 0 | 4.4464 | 0.7552 |
| packlite | 52 | 0 | 4.4464 | 0.7552 |
| days | 110 | 31 | 4.4457 | 0.4401 |
| settings | 120 | 38 | 4.4132 | 0.4116 |
| configuration | 247 | 134 | 4.3830 | 0.2606 |
| setup | 114 | 35 | 4.3629 | 0.4193 |
| allocator | 50 | 0 | 4.3601 | 0.7552 |
| feedwell | 48 | 0 | 4.2720 | 0.7552 |
| duplicate | 67 | 9 | 4.2513 | 0.5823 |
| incident | 52 | 2 | 4.2381 | 0.7001 |
| parental | 47 | 0 | 4.2272 | 0.7552 |
| byte | 72 | 12 | 4.2212 | 0.5476 |
| gb | 60 | 6 | 4.2080 | 0.6217 |
| postgresql | 60 | 6 | 4.2080 | 0.6217 |
| flapping | 46 | 0 | 4.1820 | 0.7552 |
| mkdocs | 46 | 0 | 4.1820 | 0.7552 |
| hashes | 53 | 3 | 4.1815 | 0.6759 |
| initial | 80 | 17 | 4.1804 | 0.5020 |
| specific | 204 | 106 | 4.1500 | 0.2737 |
| corkboard | 45 | 0 | 4.1363 | 0.7552 |
| anything | 116 | 41 | 4.1062 | 0.3833 |
| lines | 104 | 33 | 4.1044 | 0.4111 |
| back | 211 | 113 | 4.1020 | 0.2645 |
| bgp | 44 | 0 | 4.0901 | 0.7552 |
| genuinely | 44 | 0 | 4.0901 | 0.7552 |
| ratchet | 44 | 0 | 4.0901 | 0.7552 |
| months | 81 | 19 | 4.0794 | 0.4814 |
| within | 190 | 98 | 4.0321 | 0.2759 |
| feed | 54 | 5 | 4.0305 | 0.6305 |
| images | 129 | 52 | 4.0095 | 0.3476 |
| dash | 44 | 1 | 3.9761 | 0.7220 |
| performance | 162 | 78 | 3.9511 | 0.2965 |
| filament | 41 | 0 | 3.9481 | 0.7552 |
| sqs | 41 | 0 | 3.9481 | 0.7552 |
| took | 82 | 23 | 3.8455 | 0.4411 |
| journey | 56 | 8 | 3.8429 | 0.5729 |
| column | 83 | 24 | 3.8207 | 0.4339 |
| instances | 104 | 38 | 3.8177 | 0.3745 |
| email | 98 | 34 | 3.8114 | 0.3882 |
| hours | 98 | 34 | 3.8114 | 0.3882 |
| thistle | 38 | 0 | 3.8009 | 0.7552 |
| photos | 55 | 8 | 3.7953 | 0.5700 |
| monitoring | 93 | 31 | 3.7897 | 0.3985 |
| worth | 79 | 22 | 3.7854 | 0.4428 |
| identify | 62 | 12 | 3.7769 | 0.5204 |
| covering | 44 | 3 | 3.7541 | 0.6609 |
| wharfgate | 37 | 0 | 3.7506 | 0.7552 |
| static | 79 | 23 | 3.7167 | 0.4322 |
| dampening | 36 | 0 | 3.6995 | 0.7552 |
| handheld | 36 | 0 | 3.6995 | 0.7552 |
| nas | 36 | 0 | 3.6995 | 0.7552 |
| hook | 45 | 4 | 3.6969 | 0.6350 |
| tap | 45 | 4 | 3.6969 | 0.6350 |
| rotation | 38 | 1 | 3.6785 | 0.7169 |
| drives | 58 | 11 | 3.6725 | 0.5242 |
| due | 114 | 48 | 3.6625 | 0.3354 |
| here's | 118 | 51 | 3.6597 | 0.3279 |
| drive | 68 | 17 | 3.6574 | 0.4675 |
| efficient | 71 | 19 | 3.6449 | 0.4521 |
| query | 125 | 57 | 3.6223 | 0.3125 |

### human-favored

| n-gram | llm count | human count | z | delta |
|---|---|---|---|---|
| flux | 2 | 491 | -11.3569 | -0.6258 |
| rust | 44 | 458 | -9.3829 | -0.5029 |
| of | 2743 | 4824 | -9.3106 | -0.1241 |
| be | 618 | 1508 | -8.7013 | -0.2201 |
| in | 2258 | 3946 | -8.2681 | -0.1217 |
| https | 23 | 314 | -8.0825 | -0.5309 |
| helm | 1 | 218 | -7.5564 | -0.6247 |
| v | 29 | 295 | -7.4975 | -0.5001 |
| see | 93 | 418 | -7.0600 | -0.3690 |
| are | 849 | 1667 | -6.7614 | -0.1566 |
| gradle | 1 | 167 | -6.5978 | -0.6225 |
| as | 738 | 1477 | -6.5715 | -0.1622 |
| to | 4846 | 7098 | -6.2119 | -0.0658 |
| haskell | 0 | 144 | -6.1864 | -0.6314 |
| js | 69 | 314 | -6.1512 | -0.3715 |
| very | 34 | 229 | -6.0296 | -0.4431 |
| array | 23 | 196 | -5.9052 | -0.4776 |
| org | 13 | 168 | -5.8677 | -0.5255 |
| com | 29 | 200 | -5.6709 | -0.4468 |
| package | 23 | 185 | -5.6649 | -0.4697 |
| he | 13 | 155 | -5.5725 | -0.5174 |
| use | 358 | 777 | -5.3827 | -0.1859 |
| community | 75 | 286 | -5.3748 | -0.3330 |
| prometheus | 6 | 127 | -5.3717 | -0.5644 |
| is | 2171 | 3363 | -5.3317 | -0.0830 |
| versions | 10 | 136 | -5.3143 | -0.5303 |
| other | 153 | 422 | -5.1863 | -0.2528 |
| u | 2 | 104 | -5.0938 | -0.6032 |
| io | 2 | 101 | -5.0151 | -0.6023 |
| there | 158 | 418 | -4.9644 | -0.2416 |
| can | 732 | 1303 | -4.9109 | -0.1263 |
| d | 26 | 159 | -4.8833 | -0.4270 |
| may | 108 | 324 | -4.8771 | -0.2748 |
| components | 11 | 121 | -4.8627 | -0.5087 |
| gitops | 0 | 85 | -4.7524 | -0.6313 |
| his | 12 | 119 | -4.7384 | -0.4968 |
| linkerd | 0 | 84 | -4.7243 | -0.6313 |
| wikipedia | 0 | 84 | -4.7243 | -0.6313 |
| code | 244 | 544 | -4.6681 | -0.1936 |
| language | 15 | 123 | -4.6383 | -0.4723 |
| releases | 7 | 101 | -4.6129 | -0.5356 |
| some | 174 | 423 | -4.5727 | -0.2182 |
| have | 459 | 864 | -4.4899 | -0.1433 |
| release | 54 | 201 | -4.4432 | -0.3274 |
| react | 3 | 83 | -4.4241 | -0.5794 |
| html | 14 | 108 | -4.2858 | -0.4636 |
| bazel | 0 | 69 | -4.2816 | -0.6313 |
| lts | 1 | 72 | -4.2755 | -0.6108 |
| node | 129 | 330 | -4.2646 | -0.2323 |
| series | 12 | 102 | -4.2567 | -0.4772 |
| used | 141 | 348 | -4.2166 | -0.2224 |
| also | 195 | 436 | -4.1934 | -0.1943 |
| they | 228 | 488 | -4.1780 | -0.1816 |
| http | 28 | 136 | -4.1601 | -0.3845 |
| meetup | 0 | 64 | -4.1235 | -0.6313 |
| apache | 4 | 76 | -4.1171 | -0.5572 |
| will | 555 | 966 | -4.0065 | -0.1191 |
| controller | 7 | 80 | -3.9773 | -0.5128 |
| packages | 6 | 77 | -3.9678 | -0.5247 |
| kubernetes | 49 | 172 | -3.9672 | -0.3136 |
| class | 24 | 120 | -3.9533 | -0.3902 |
| core | 38 | 149 | -3.9377 | -0.3392 |
| x | 59 | 190 | -3.9368 | -0.2926 |
| vulnerable | 0 | 58 | -3.9254 | -0.6313 |
| operator | 3 | 67 | -3.9174 | -0.5677 |
| all | 404 | 738 | -3.9031 | -0.1340 |
| docs | 8 | 79 | -3.8575 | -0.4963 |
| en | 1 | 59 | -3.8507 | -0.6064 |
| functions | 30 | 129 | -3.8392 | -0.3594 |
| many | 75 | 215 | -3.8265 | -0.2629 |
| component | 5 | 69 | -3.7917 | -0.5315 |
| called | 27 | 121 | -3.7902 | -0.3681 |
| cncf | 0 | 54 | -3.7876 | -0.6313 |
| function | 58 | 182 | -3.7813 | -0.2861 |
| t | 3 | 63 | -3.7804 | -0.5638 |
| github | 93 | 244 | -3.7602 | -0.2392 |
| rfcs | 0 | 53 | -3.7524 | -0.6313 |
| framework | 11 | 83 | -3.7371 | -0.4603 |
| func | 0 | 52 | -3.7168 | -0.6313 |
| apps | 1 | 55 | -3.7103 | -0.6046 |
| want | 118 | 284 | -3.7028 | -0.2153 |
| repositories | 3 | 60 | -3.6744 | -0.5607 |
| object | 53 | 168 | -3.6598 | -0.2886 |
| license | 34 | 131 | -3.6569 | -0.3352 |
| openssl | 0 | 50 | -3.6446 | -0.6313 |
| learn | 17 | 94 | -3.6318 | -0.4091 |
| swift | 6 | 67 | -3.6266 | -0.5103 |
| selfie | 0 | 49 | -3.6080 | -0.6313 |
| arrays | 2 | 55 | -3.5998 | -0.5791 |
| education | 0 | 48 | -3.5710 | -0.6313 |
| examples | 14 | 85 | -3.5618 | -0.4257 |
| program | 35 | 129 | -3.5384 | -0.3250 |
| flagger | 0 | 47 | -3.5335 | -0.6313 |
| libcloud | 0 | 47 | -3.5335 | -0.6313 |
| vue | 0 | 47 | -3.5335 | -0.6313 |
| property | 9 | 72 | -3.5287 | -0.4688 |
| ref | 3 | 56 | -3.5283 | -0.5559 |
| unity | 0 | 46 | -3.4957 | -0.6313 |
| view | 18 | 92 | -3.4909 | -0.3944 |
| exposure | 3 | 55 | -3.4909 | -0.5546 |
| craft | 2 | 52 | -3.4879 | -0.5762 |
| svg | 2 | 52 | -3.4879 | -0.5762 |
| chinese | 0 | 45 | -3.4575 | -0.6313 |
| oauthlib | 0 | 45 | -3.4575 | -0.6313 |
| users | 56 | 165 | -3.4288 | -0.2700 |
| module | 28 | 111 | -3.4183 | -0.3416 |
| method | 30 | 115 | -3.4165 | -0.3341 |
| lens | 1 | 47 | -3.4123 | -0.6002 |
| contributors | 6 | 61 | -3.4070 | -0.4997 |
| np | 0 | 43 | -3.3798 | -0.6313 |
| states | 3 | 52 | -3.3761 | -0.5505 |
| expression | 10 | 70 | -3.3672 | -0.4490 |
| apis | 3 | 51 | -3.3371 | -0.5490 |
| binary | 18 | 87 | -3.3201 | -0.3834 |
| decorators | 0 | 41 | -3.3003 | -0.6313 |
| interpreter | 0 | 41 | -3.3003 | -0.6313 |
| sparkplug | 0 | 41 | -3.3003 | -0.6313 |
| de | 2 | 47 | -3.2932 | -0.5706 |
| babel | 0 | 40 | -3.2598 | -0.6312 |
| disposition | 0 | 40 | -3.2598 | -0.6312 |

## 2-gram

Total n-gram tokens: llm=156128, human=180490. Scored vocabulary (>= min-count): 10569.

### llm-favored

| n-gram | llm count | human count | z | delta |
|---|---|---|---|---|
| rather than | 155 | 42 | 5.1981 | 0.4357 |
| perceptual hashing | 70 | 0 | 5.0658 | 0.7416 |
| parental leave | 47 | 0 | 4.1507 | 0.7415 |
| your name | 49 | 1 | 4.1288 | 0.7117 |
| is crucial | 44 | 1 | 3.9008 | 0.7083 |
| the migration | 53 | 5 | 3.8973 | 0.6148 |
| for our | 65 | 11 | 3.8966 | 0.5312 |
| configuration file | 57 | 7 | 3.8896 | 0.5815 |
| front matter | 42 | 1 | 3.8058 | 0.7068 |
| our team | 51 | 5 | 3.8041 | 0.6103 |
| bump allocator | 39 | 0 | 3.7809 | 0.7415 |
| for your | 115 | 45 | 3.7083 | 0.3422 |
| memory file | 37 | 0 | 3.6826 | 0.7415 |
| github pull | 36 | 0 | 3.6325 | 0.7415 |
| spot instances | 35 | 0 | 3.5817 | 0.7415 |
| the file | 52 | 8 | 3.5603 | 0.5473 |
| line tool | 37 | 1 | 3.5572 | 0.7023 |
| root cause | 37 | 1 | 3.5572 | 0.7023 |
| this email | 34 | 0 | 3.5301 | 0.7415 |
| ensure that | 57 | 11 | 3.5290 | 0.5072 |
| photo library | 33 | 0 | 3.4778 | 0.7415 |
| the bed | 31 | 0 | 3.3708 | 0.7415 |
| and a | 172 | 96 | 3.3601 | 0.2381 |
| of your | 109 | 49 | 3.2711 | 0.3029 |
| the nozzle | 29 | 0 | 3.2602 | 0.7415 |
| six months | 36 | 3 | 3.2593 | 0.6281 |
| designed to | 68 | 21 | 3.2502 | 0.4041 |
| by following | 31 | 1 | 3.2340 | 0.6949 |
| best regards | 28 | 0 | 3.2035 | 0.7415 |
| the necessary | 30 | 1 | 3.1770 | 0.6935 |
| you through | 32 | 2 | 3.1587 | 0.6545 |
| handheld transceiver | 27 | 0 | 3.1458 | 0.7415 |
| to your | 117 | 58 | 3.1205 | 0.2741 |
| and ensure | 29 | 1 | 3.1190 | 0.6919 |
| crucial for | 29 | 1 | 3.1190 | 0.6919 |
| making it | 50 | 12 | 3.0892 | 0.4628 |
| setting up | 50 | 12 | 3.0892 | 0.4628 |
| can lead | 28 | 1 | 3.0598 | 0.6902 |
| covering index | 25 | 0 | 3.0270 | 0.7415 |
| free list | 25 | 0 | 3.0270 | 0.7415 |
| static site | 27 | 1 | 2.9995 | 0.6884 |
| here's a | 41 | 8 | 2.9838 | 0.5051 |
| on your | 91 | 41 | 2.9833 | 0.3023 |
| channel memories | 24 | 0 | 2.9658 | 0.7415 |
| following these | 24 | 0 | 2.9658 | 0.7415 |
| hamming distance | 24 | 0 | 2.9658 | 0.7415 |
| your handheld | 24 | 0 | 2.9658 | 0.7415 |
| this will | 75 | 30 | 2.9515 | 0.3360 |
| and i | 79 | 33 | 2.9407 | 0.3239 |
| the app | 45 | 11 | 2.9116 | 0.4588 |
| leading to | 30 | 3 | 2.9098 | 0.6079 |
| a basic | 34 | 5 | 2.9058 | 0.5546 |
| aws bill | 23 | 0 | 2.9034 | 0.7415 |
| diagnostic steps | 23 | 0 | 2.9034 | 0.7415 |
| what got | 23 | 0 | 2.9034 | 0.7415 |
| your specific | 25 | 1 | 2.8752 | 0.6843 |
| the index | 35 | 6 | 2.8502 | 0.5289 |
| the engine | 29 | 3 | 2.8476 | 0.6038 |
| building a | 31 | 4 | 2.8440 | 0.5744 |
| the part | 31 | 4 | 2.8440 | 0.5744 |
| and i'd | 22 | 0 | 2.8395 | 0.7415 |
| monitoring tools | 22 | 0 | 2.8395 | 0.7415 |
| six weeks | 22 | 0 | 2.8395 | 0.7415 |
| the dash | 22 | 0 | 2.8395 | 0.7415 |
| webhook receiver | 22 | 0 | 2.8395 | 0.7415 |
| and the | 299 | 225 | 2.8257 | 0.1427 |
| a significant | 36 | 7 | 2.7986 | 0.5057 |
| a lightweight | 26 | 2 | 2.7934 | 0.6361 |
| your site | 26 | 2 | 2.7934 | 0.6361 |
| autoplaying videos | 21 | 0 | 2.7743 | 0.7414 |
| duplicate photos | 21 | 0 | 2.7743 | 0.7414 |
| food bank | 21 | 0 | 2.7743 | 0.7414 |
| layer shifting | 21 | 0 | 2.7743 | 0.7414 |
| leave policy | 21 | 0 | 2.7743 | 0.7414 |
| read replica | 21 | 0 | 2.7743 | 0.7414 |
| stage builds | 21 | 0 | 2.7743 | 0.7414 |
| walk you | 23 | 1 | 2.7453 | 0.6796 |
| me to | 35 | 7 | 2.7377 | 0.5003 |
| this guide | 40 | 10 | 2.7227 | 0.4538 |
| the primary | 27 | 3 | 2.7195 | 0.5949 |
| to our | 63 | 25 | 2.7192 | 0.3382 |
| commit hook | 20 | 0 | 2.7074 | 0.7414 |
| gb of | 20 | 0 | 2.7074 | 0.7414 |
| jupyter environment | 20 | 0 | 2.7074 | 0.7414 |
| lifetime elision | 20 | 0 | 2.7074 | 0.7414 |
| using perceptual | 20 | 0 | 2.7074 | 0.7414 |
| set up | 46 | 14 | 2.6914 | 0.4077 |
| hope this | 22 | 1 | 2.6781 | 0.6769 |
| ideal for | 22 | 1 | 2.6781 | 0.6769 |
| more robust | 22 | 1 | 2.6781 | 0.6769 |
| to its | 44 | 13 | 2.6682 | 0.4150 |
| these steps | 24 | 2 | 2.6611 | 0.6281 |
| let me | 26 | 3 | 2.6533 | 0.5899 |
| that's the | 28 | 4 | 2.6522 | 0.5591 |
| a bump | 19 | 0 | 2.6388 | 0.7414 |
| amazon sqs | 19 | 0 | 2.6388 | 0.7414 |
| finds you | 19 | 0 | 2.6388 | 0.7414 |
| from gerrit | 19 | 0 | 2.6388 | 0.7414 |
| route flapping | 19 | 0 | 2.6388 | 0.7414 |
| your transceiver | 19 | 0 | 2.6388 | 0.7414 |
| to ensure | 68 | 30 | 2.6210 | 0.3083 |
| against a | 21 | 1 | 2.6091 | 0.6740 |
| configuration files | 21 | 1 | 2.6091 | 0.6740 |
| importance of | 21 | 1 | 2.6091 | 0.6740 |
| the print | 21 | 1 | 2.6091 | 0.6740 |
| a job | 23 | 2 | 2.5925 | 0.6237 |
| a robust | 23 | 2 | 2.5925 | 0.6237 |
| adding a | 29 | 5 | 2.5908 | 0.5279 |
| you for | 29 | 5 | 2.5908 | 0.5279 |
| this command | 27 | 4 | 2.5855 | 0.5534 |
| on our | 44 | 14 | 2.5768 | 0.3966 |
| a covering | 18 | 0 | 2.5684 | 0.7414 |
| and select | 18 | 0 | 2.5684 | 0.7414 |
| backup script | 18 | 0 | 2.5684 | 0.7414 |
| my first | 18 | 0 | 2.5684 | 0.7414 |
| route dampening | 18 | 0 | 2.5684 | 0.7414 |
| site generator | 18 | 0 | 2.5684 | 0.7414 |
| you well | 18 | 0 | 2.5684 | 0.7414 |
| the whole | 60 | 25 | 2.5671 | 0.3246 |
| thank you | 42 | 13 | 2.5514 | 0.4035 |

### human-favored

| n-gram | llm count | human count | z | delta |
|---|---|---|---|---|
| of the | 491 | 1151 | -7.7345 | -0.2224 |
| it is | 88 | 311 | -5.5939 | -0.3291 |
| in the | 523 | 966 | -5.0347 | -0.1514 |
| have to | 8 | 109 | -4.8817 | -0.5441 |
| you are | 28 | 157 | -4.8747 | -0.4255 |
| versions of | 3 | 85 | -4.5895 | -0.5943 |
| for example | 48 | 183 | -4.4751 | -0.3466 |
| can be | 121 | 313 | -4.4461 | -0.2492 |
| the flux | 0 | 71 | -4.4377 | -0.6450 |
| be used | 11 | 100 | -4.3938 | -0.4997 |
| there are | 36 | 150 | -4.2360 | -0.3664 |
| in rust | 6 | 82 | -4.2354 | -0.5444 |
| want to | 70 | 208 | -4.0725 | -0.2860 |
| of node | 1 | 62 | -4.0421 | -0.6213 |
| all versions | 0 | 58 | -4.0107 | -0.6450 |
| if you | 229 | 460 | -3.9883 | -0.1766 |
| as well | 15 | 96 | -3.9677 | -0.4483 |
| flux v | 0 | 55 | -3.9056 | -0.6450 |
| in new | 0 | 54 | -3.8699 | -0.6450 |
| time series | 0 | 52 | -3.7976 | -0.6450 |
| d array | 0 | 51 | -3.7609 | -0.6450 |
| this week | 9 | 75 | -3.7394 | -0.4882 |
| to use | 56 | 169 | -3.7132 | -0.2899 |
| to be | 134 | 295 | -3.6218 | -0.2035 |
| week in | 1 | 50 | -3.6074 | -0.6157 |
| see the | 16 | 87 | -3.5907 | -0.4198 |
| opens in | 0 | 46 | -3.5717 | -0.6450 |
| to the | 345 | 601 | -3.5286 | -0.1330 |
| active record | 0 | 44 | -3.4932 | -0.6450 |
| may be | 11 | 71 | -3.4200 | -0.4497 |
| core dump | 0 | 42 | -3.4128 | -0.6450 |
| rust community | 0 | 42 | -3.4128 | -0.6450 |
| is to | 21 | 91 | -3.3590 | -0.3747 |
| support for | 16 | 80 | -3.3410 | -0.4038 |
| to have | 9 | 64 | -3.3311 | -0.4651 |
| open source | 4 | 51 | -3.3106 | -0.5377 |
| the package | 2 | 45 | -3.2896 | -0.5818 |
| helm v | 0 | 39 | -3.2887 | -0.6450 |
| share on | 0 | 39 | -3.2887 | -0.6450 |
| possible to | 1 | 42 | -3.2860 | -0.6103 |
| all the | 28 | 102 | -3.2576 | -0.3360 |
| new window | 3 | 47 | -3.2564 | -0.5562 |
| check out | 1 | 41 | -3.2436 | -0.6095 |
| they are | 29 | 103 | -3.2261 | -0.3301 |
| note that | 13 | 70 | -3.2099 | -0.4179 |
| open access | 0 | 37 | -3.2032 | -0.6450 |
| the rust | 0 | 36 | -3.1596 | -0.6450 |
| that you | 15 | 72 | -3.1198 | -0.3958 |
| join the | 0 | 35 | -3.1154 | -0.6450 |
| ways to | 4 | 46 | -3.0992 | -0.5272 |
| it will | 23 | 87 | -3.0715 | -0.3447 |
| event time | 0 | 34 | -3.0706 | -0.6450 |
| will be | 99 | 216 | -3.0630 | -0.2008 |
| the error | 4 | 44 | -3.0108 | -0.5224 |
| is that | 23 | 84 | -2.9603 | -0.3366 |
| so that | 7 | 50 | -2.9475 | -0.4658 |
| more information | 11 | 59 | -2.9428 | -0.4172 |
| important to | 1 | 34 | -2.9299 | -0.6024 |
| are available | 1 | 33 | -2.8824 | -0.6011 |
| to get | 45 | 121 | -2.8615 | -0.2595 |
| you can | 274 | 460 | -2.8370 | -0.1213 |
| in fact | 1 | 32 | -2.8340 | -0.5998 |
| order to | 6 | 45 | -2.8294 | -0.4731 |
| the community | 6 | 45 | -2.8294 | -0.4731 |
| to open | 4 | 40 | -2.8265 | -0.5114 |
| set of | 13 | 60 | -2.8028 | -0.3878 |
| bug scrub | 0 | 28 | -2.7865 | -0.6449 |
| flux and | 0 | 28 | -2.7865 | -0.6449 |
| usa by | 0 | 28 | -2.7865 | -0.6449 |
| there is | 44 | 117 | -2.7859 | -0.2565 |
| the license | 1 | 31 | -2.7848 | -0.5984 |
| well as | 10 | 53 | -2.7775 | -0.4149 |
| is not | 67 | 154 | -2.7549 | -0.2158 |
| is used | 9 | 50 | -2.7415 | -0.4237 |
| is important | 0 | 27 | -2.7362 | -0.6449 |
| right to | 0 | 27 | -2.7362 | -0.6449 |
| are very | 1 | 30 | -2.7348 | -0.5969 |
| by the | 70 | 158 | -2.7314 | -0.2106 |
| that it | 15 | 62 | -2.7121 | -0.3645 |
| the public | 3 | 35 | -2.7089 | -0.5287 |
| the case | 2 | 32 | -2.6928 | -0.5579 |
| that they | 6 | 42 | -2.6877 | -0.4626 |
| build script | 0 | 26 | -2.6851 | -0.6449 |
| of flux | 0 | 26 | -2.6851 | -0.6449 |
| of helm | 0 | 26 | -2.6851 | -0.6449 |
| to flux | 0 | 26 | -2.6851 | -0.6449 |
| a set | 4 | 37 | -2.6809 | -0.5018 |
| for more | 35 | 98 | -2.6658 | -0.2703 |
| a very | 9 | 48 | -2.6490 | -0.4161 |
| to learn | 9 | 48 | -2.6490 | -0.4161 |
| core team | 0 | 25 | -2.6329 | -0.6449 |
| united states | 0 | 25 | -2.6329 | -0.6449 |
| click to | 1 | 28 | -2.6319 | -0.5936 |
| you may | 16 | 62 | -2.6260 | -0.3502 |
| used to | 25 | 79 | -2.6227 | -0.3015 |
| a number | 9 | 47 | -2.6018 | -0.4122 |
| is an | 27 | 82 | -2.5977 | -0.2915 |
| be found | 2 | 30 | -2.5893 | -0.5526 |
| comment period | 0 | 24 | -2.5797 | -0.6449 |
| public domain | 0 | 24 | -2.5797 | -0.6449 |
| a look | 1 | 27 | -2.5789 | -0.5918 |
| of a | 105 | 206 | -2.5698 | -0.1693 |
| would like | 12 | 52 | -2.5388 | -0.3747 |
| are looking | 2 | 29 | -2.5361 | -0.5496 |
| js v | 0 | 23 | -2.5254 | -0.6449 |
| the helm | 0 | 23 | -2.5254 | -0.6449 |
| in addition | 1 | 26 | -2.5248 | -0.5899 |
| is very | 1 | 26 | -2.5248 | -0.5899 |
| new to | 1 | 26 | -2.5248 | -0.5899 |
| to make | 35 | 94 | -2.5192 | -0.2591 |
| when using | 3 | 31 | -2.5020 | -0.5153 |
| list of | 24 | 74 | -2.4948 | -0.2953 |
| should be | 45 | 110 | -2.4903 | -0.2333 |
| command line | 6 | 38 | -2.4884 | -0.4465 |
| new features | 4 | 33 | -2.4749 | -0.4867 |
| work on | 4 | 33 | -2.4749 | -0.4867 |
| and to | 7 | 40 | -2.4747 | -0.4287 |
| final comment | 0 | 22 | -2.4699 | -0.6449 |
| functional programming | 0 | 22 | -2.4699 | -0.6449 |
| the united | 0 | 22 | -2.4699 | -0.6449 |

## 3-gram

Total n-gram tokens: llm=127706, human=147005. Scored vocabulary (>= min-count): 3295.

### llm-favored

| n-gram | llm count | human count | z | delta |
|---|---|---|---|---|
| github pull requests | 36 | 0 | 3.6257 | 0.7401 |
| can lead to | 28 | 1 | 3.0536 | 0.6888 |
| by following these | 24 | 0 | 2.9602 | 0.7401 |
| parental leave policy | 21 | 0 | 2.7690 | 0.7400 |
| you for your | 21 | 0 | 2.7690 | 0.7400 |
| walk you through | 23 | 1 | 2.7396 | 0.6782 |
| is crucial for | 20 | 0 | 2.7023 | 0.7400 |
| to reach out | 19 | 0 | 2.6338 | 0.7400 |
| using perceptual hashing | 19 | 0 | 2.6338 | 0.7400 |
| your handheld transceiver | 19 | 0 | 2.6338 | 0.7400 |
| the importance of | 21 | 1 | 2.6037 | 0.6726 |
| thank you for | 25 | 3 | 2.5793 | 0.5832 |
| a bump allocator | 18 | 0 | 2.5636 | 0.7400 |
| a covering index | 18 | 0 | 2.5636 | 0.7400 |
| finds you well | 18 | 0 | 2.5636 | 0.7400 |
| static site generator | 18 | 0 | 2.5636 | 0.7400 |
| to github pull | 17 | 0 | 2.4913 | 0.7400 |
| i hope this | 19 | 1 | 2.4603 | 0.6659 |
| is a powerful | 21 | 2 | 2.4444 | 0.6122 |
| from gerrit to | 16 | 0 | 2.4169 | 0.7400 |
| gerrit to github | 16 | 0 | 2.4169 | 0.7400 |
| javascript and css | 16 | 0 | 2.4169 | 0.7400 |
| new read replica | 16 | 0 | 2.4169 | 0.7400 |
| with the actual | 16 | 0 | 2.4169 | 0.7400 |
| the root cause | 18 | 1 | 2.3855 | 0.6621 |
| the following command | 24 | 4 | 2.3696 | 0.5324 |
| hesitate to reach | 15 | 0 | 2.3402 | 0.7400 |
| feel free to | 30 | 8 | 2.2931 | 0.4378 |
| is the part | 14 | 0 | 2.2608 | 0.7400 |
| the lifetime of | 14 | 0 | 2.2608 | 0.7400 |
| the migration script | 14 | 0 | 2.2608 | 0.7400 |
| what got easier | 14 | 0 | 2.2608 | 0.7400 |
| your photo library | 14 | 0 | 2.2608 | 0.7400 |
| when dealing with | 16 | 1 | 2.2286 | 0.6530 |
| you through the | 16 | 1 | 2.2286 | 0.6530 |
| add the following | 27 | 7 | 2.1996 | 0.4442 |
| back to a | 13 | 0 | 2.1786 | 0.7400 |
| bed adhesion failures | 13 | 0 | 2.1786 | 0.7400 |
| ensure you have | 13 | 0 | 2.1786 | 0.7400 |
| following these steps | 13 | 0 | 2.1786 | 0.7400 |
| i'm excited to | 13 | 0 | 2.1786 | 0.7400 |
| ratchet task queue | 13 | 0 | 2.1786 | 0.7400 |
| zero or more | 13 | 0 | 2.1786 | 0.7400 |
| based on your | 15 | 1 | 2.1460 | 0.6477 |
| due to its | 15 | 1 | 2.1460 | 0.6477 |
| ensure that the | 17 | 2 | 2.1337 | 0.5859 |
| backed jupyter environment | 12 | 0 | 2.0931 | 0.7400 |
| email finds you | 12 | 0 | 2.0931 | 0.7400 |
| food bank name | 12 | 0 | 2.0931 | 0.7400 |
| hope this email | 12 | 0 | 2.0931 | 0.7400 |
| multiple comparisons problem | 12 | 0 | 2.0931 | 0.7400 |
| our aws bill | 12 | 0 | 2.0931 | 0.7400 |
| the migration process | 12 | 0 | 2.0931 | 0.7400 |
| this email finds | 12 | 0 | 2.0931 | 0.7400 |
| walnut dining table | 12 | 0 | 2.0931 | 0.7400 |
| this command will | 14 | 1 | 2.0602 | 0.6416 |
| will walk you | 14 | 1 | 2.0602 | 0.6416 |
| you have any | 25 | 7 | 2.0518 | 0.4264 |
| a composite index | 11 | 0 | 2.0040 | 0.7400 |
| allowed us to | 11 | 0 | 2.0040 | 0.7400 |
| choosing the right | 11 | 0 | 2.0040 | 0.7400 |
| dive into the | 11 | 0 | 2.0040 | 0.7400 |
| is the one | 11 | 0 | 2.0040 | 0.7400 |
| lifetime elision rules | 11 | 0 | 2.0040 | 0.7400 |
| like me to | 11 | 0 | 2.0040 | 0.7400 |
| of the job | 11 | 0 | 2.0040 | 0.7400 |
| please don't hesitate | 11 | 0 | 2.0040 | 0.7400 |
| that's the whole | 11 | 0 | 2.0040 | 0.7400 |
| this guide will | 11 | 0 | 2.0040 | 0.7400 |
| you like me | 11 | 0 | 2.0040 | 0.7400 |
| a testament to | 13 | 1 | 1.9708 | 0.6347 |
| through the process | 13 | 1 | 1.9708 | 0.6347 |
| to access the | 13 | 1 | 1.9708 | 0.6347 |
| here's an example | 22 | 6 | 1.9459 | 0.4325 |
| a simple webhook | 10 | 0 | 1.9107 | 0.7400 |
| and css files | 10 | 0 | 1.9107 | 0.7400 |
| for your continued | 10 | 0 | 1.9107 | 0.7400 |
| front matter overrides | 10 | 0 | 1.9107 | 0.7400 |
| guide will walk | 10 | 0 | 1.9107 | 0.7400 |
| mobile app redesign | 10 | 0 | 1.9107 | 0.7400 |
| multiple log files | 10 | 0 | 1.9107 | 0.7400 |
| nonprofit food bank | 10 | 0 | 1.9107 | 0.7400 |
| page front matter | 10 | 0 | 1.9107 | 0.7400 |
| payment processing api | 10 | 0 | 1.9107 | 0.7400 |
| pixel quest adventures | 10 | 0 | 1.9107 | 0.7400 |
| reply to this | 10 | 0 | 1.9107 | 0.7400 |
| simple webhook receiver | 10 | 0 | 1.9107 | 0.7400 |
| systemd unit file | 10 | 0 | 1.9107 | 0.7400 |
| the multiple comparisons | 10 | 0 | 1.9107 | 0.7400 |
| the new read | 10 | 0 | 1.9107 | 0.7400 |
| to amazon sqs | 10 | 0 | 1.9107 | 0.7400 |
| verify that the | 10 | 0 | 1.9107 | 0.7400 |
| don't hesitate to | 16 | 3 | 1.8788 | 0.5112 |
| what you need | 12 | 1 | 1.8774 | 0.6267 |
| let me know | 14 | 2 | 1.8706 | 0.5577 |
| to set up | 14 | 2 | 1.8706 | 0.5577 |
| have any questions | 19 | 5 | 1.8344 | 0.4408 |
| a free list | 9 | 0 | 1.8127 | 0.7400 |
| app store review | 9 | 0 | 1.8127 | 0.7400 |
| are committed to | 9 | 0 | 1.8127 | 0.7400 |
| automatic database backups | 9 | 0 | 1.8127 | 0.7400 |
| bump allocator with | 9 | 0 | 1.8127 | 0.7400 |
| by the end | 9 | 0 | 1.8127 | 0.7400 |
| detailed information about | 9 | 0 | 1.8127 | 0.7400 |
| directory for your | 9 | 0 | 1.8127 | 0.7400 |
| do it again | 9 | 0 | 1.8127 | 0.7400 |
| for our team | 9 | 0 | 1.8127 | 0.7400 |
| from a backup | 9 | 0 | 1.8127 | 0.7400 |
| from confluence to | 9 | 0 | 1.8127 | 0.7400 |
| full table scan | 9 | 0 | 1.8127 | 0.7400 |
| instances for batch | 9 | 0 | 1.8127 | 0.7400 |
| it ideal for | 9 | 0 | 1.8127 | 0.7400 |
| layer shifting mid | 9 | 0 | 1.8127 | 0.7400 |
| let's dive into | 9 | 0 | 1.8127 | 0.7400 |
| library designed to | 9 | 0 | 1.8127 | 0.7400 |
| line tool for | 9 | 0 | 1.8127 | 0.7400 |
| line tool in | 9 | 0 | 1.8127 | 0.7400 |
| mobile distribution route | 9 | 0 | 1.8127 | 0.7400 |
| out to our | 9 | 0 | 1.8127 | 0.7400 |
| spot instances for | 9 | 0 | 1.8127 | 0.7400 |

### human-favored

| n-gram | llm count | human count | z | delta |
|---|---|---|---|---|
| if you are | 9 | 96 | -4.4389 | -0.5205 |
| all versions of | 0 | 58 | -4.0197 | -0.6464 |
| versions of node | 0 | 53 | -3.8425 | -0.6464 |
| this week in | 0 | 49 | -3.6946 | -0.6464 |
| in new window | 0 | 47 | -3.6184 | -0.6464 |
| opens in new | 0 | 46 | -3.5796 | -0.6464 |
| week in rust | 0 | 44 | -3.5009 | -0.6464 |
| a number of | 1 | 41 | -3.2512 | -0.6109 |
| for more information | 7 | 51 | -3.0005 | -0.4702 |
| in order to | 5 | 45 | -2.9497 | -0.4998 |
| as well as | 8 | 52 | -2.9417 | -0.4522 |
| a set of | 3 | 36 | -2.7659 | -0.5330 |
| you want to | 28 | 85 | -2.6570 | -0.2928 |
| can be used | 10 | 48 | -2.5563 | -0.3972 |
| click to share | 0 | 23 | -2.5310 | -0.6464 |
| to share on | 0 | 23 | -2.5310 | -0.6464 |
| can be found | 1 | 26 | -2.5309 | -0.5913 |
| if you want | 20 | 67 | -2.5218 | -0.3174 |
| would like to | 9 | 45 | -2.5143 | -0.4052 |
| a look at | 0 | 22 | -2.4753 | -0.6464 |
| final comment period | 0 | 22 | -2.4753 | -0.6464 |
| you have to | 1 | 24 | -2.4190 | -0.5869 |
| it is possible | 0 | 21 | -2.4184 | -0.6464 |
| the united states | 0 | 21 | -2.4184 | -0.6464 |
| take a look | 1 | 23 | -2.3611 | -0.5845 |
| of the week | 3 | 28 | -2.3426 | -0.5044 |
| be used to | 4 | 30 | -2.3170 | -0.4745 |
| all of the | 1 | 22 | -2.3018 | -0.5818 |
| that you can | 1 | 22 | -2.3018 | -0.5818 |
| right to education | 0 | 19 | -2.3004 | -0.6463 |
| rust community team | 0 | 19 | -2.3004 | -0.6463 |
| the right to | 0 | 19 | -2.3004 | -0.6463 |
| we are very | 0 | 19 | -2.3004 | -0.6463 |
| is possible to | 0 | 18 | -2.2390 | -0.6463 |
| note that the | 0 | 18 | -2.2390 | -0.6463 |
| the case of | 0 | 18 | -2.2390 | -0.6463 |
| be able to | 13 | 47 | -2.2107 | -0.3356 |
| you would like | 0 | 17 | -2.1759 | -0.6463 |
| there is a | 5 | 29 | -2.1241 | -0.4328 |
| the same time | 1 | 19 | -2.1142 | -0.5722 |
| denial of service | 0 | 16 | -2.1109 | -0.6463 |
| is a weekly | 0 | 16 | -2.1109 | -0.6463 |
| the circuit breaker | 0 | 16 | -2.1109 | -0.6463 |
| tweet us at | 0 | 16 | -2.1109 | -0.6463 |
| updates from rust | 0 | 16 | -2.1109 | -0.6463 |
| in addition to | 1 | 18 | -2.0479 | -0.5684 |
| that it is | 1 | 18 | -2.0479 | -0.5684 |
| a weekly summary | 0 | 15 | -2.0439 | -0.6463 |
| call for participation | 0 | 15 | -2.0439 | -0.6463 |
| oh my bash | 0 | 15 | -2.0439 | -0.6463 |
| the public domain | 0 | 15 | -2.0439 | -0.6463 |
| these are the | 0 | 15 | -2.0439 | -0.6463 |
| to work on | 0 | 15 | -2.0439 | -0.6463 |
| weekly summary of | 0 | 15 | -2.0439 | -0.6463 |
| if you like | 0 | 14 | -1.9746 | -0.6463 |
| in the case | 0 | 14 | -1.9746 | -0.6463 |
| team meeting at | 0 | 14 | -1.9746 | -0.6463 |
| the core team | 0 | 14 | -1.9746 | -0.6463 |
| the flux community | 0 | 14 | -1.9746 | -0.6463 |
| looking forward to | 1 | 16 | -1.9089 | -0.5593 |
| to do that | 1 | 16 | -1.9089 | -0.5593 |
| to make sure | 1 | 16 | -1.9089 | -0.5593 |
| are available for | 0 | 13 | -1.9028 | -0.6463 |
| are going to | 0 | 13 | -1.9028 | -0.6463 |
| check out the | 0 | 13 | -1.9028 | -0.6463 |
| flux works with | 0 | 13 | -1.9028 | -0.6463 |
| in the flux | 0 | 13 | -1.9028 | -0.6463 |
| in the past | 0 | 13 | -1.9028 | -0.6463 |
| is important to | 0 | 13 | -1.9028 | -0.6463 |
| is not the | 0 | 13 | -1.9028 | -0.6463 |
| it is important | 0 | 13 | -1.9028 | -0.6463 |
| of the box | 0 | 13 | -1.9028 | -0.6463 |
| that they are | 0 | 13 | -1.9028 | -0.6463 |
| the most important | 0 | 13 | -1.9028 | -0.6463 |
| the needs of | 0 | 13 | -1.9028 | -0.6463 |
| the rust community | 0 | 13 | -1.9028 | -0.6463 |
| to be able | 0 | 13 | -1.9028 | -0.6463 |
| to flux v | 0 | 13 | -1.9028 | -0.6463 |
| you are a | 0 | 13 | -1.9028 | -0.6463 |
| to get involved | 5 | 25 | -1.8739 | -0.4052 |
| to contribute to | 2 | 18 | -1.8654 | -0.4997 |
| another issue of | 1 | 15 | -1.8355 | -0.5540 |
| it's important to | 1 | 15 | -1.8355 | -0.5540 |
| provided by the | 1 | 15 | -1.8355 | -0.5540 |
| to another issue | 1 | 15 | -1.8355 | -0.5540 |
| welcome to another | 1 | 15 | -1.8355 | -0.5540 |
| you can read | 1 | 15 | -1.8355 | -0.5540 |
| enter image description | 0 | 12 | -1.8281 | -0.6463 |
| for the suggestion | 0 | 12 | -1.8281 | -0.6463 |
| have to be | 0 | 12 | -1.8281 | -0.6463 |
| image description here | 0 | 12 | -1.8281 | -0.6463 |
| let's recap what | 0 | 12 | -1.8281 | -0.6463 |
| look at our | 0 | 12 | -1.8281 | -0.6463 |
| talk to us | 0 | 12 | -1.8281 | -0.6463 |
| the flux family | 0 | 12 | -1.8281 | -0.6463 |
| to get the | 0 | 12 | -1.8281 | -0.6463 |
| under the apache | 0 | 12 | -1.8281 | -0.6463 |
| up to date | 0 | 12 | -1.8281 | -0.6463 |
| ways to do | 0 | 12 | -1.8281 | -0.6463 |
| the following example | 1 | 14 | -1.7593 | -0.5479 |
| want to get | 1 | 14 | -1.7593 | -0.5479 |
| will not be | 1 | 14 | -1.7593 | -0.5479 |
| about what has | 0 | 11 | -1.7503 | -0.6463 |
| are looking forward | 0 | 11 | -1.7503 | -0.6463 |
| checks that the | 0 | 11 | -1.7503 | -0.6463 |
| community team meeting | 0 | 11 | -1.7503 | -0.6463 |
| give us feedback | 0 | 11 | -1.7503 | -0.6463 |
| is easy to | 0 | 11 | -1.7503 | -0.6463 |
| is part of | 0 | 11 | -1.7503 | -0.6463 |
| more about the | 0 | 11 | -1.7503 | -0.6463 |
| new to flux | 0 | 11 | -1.7503 | -0.6463 |
| the build script | 0 | 11 | -1.7503 | -0.6463 |
| the purpose of | 0 | 11 | -1.7503 | -0.6463 |
| to see our | 0 | 11 | -1.7503 | -0.6463 |
| to us in | 0 | 11 | -1.7503 | -0.6463 |
| a lot of | 25 | 58 | -1.7193 | -0.2199 |
| a variety of | 1 | 13 | -1.6799 | -0.5410 |
| copy of the | 1 | 13 | -1.6799 | -0.5410 |
| make sure that | 1 | 13 | -1.6799 | -0.5410 |
| new features and | 1 | 13 | -1.6799 | -0.5410 |
