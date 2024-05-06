#!/bin/bash

for data in BioASQ
do
  for v_size in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000
  do
    for frac in 0 #0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 5 5.25 5.5 5.75 6 6.25 6.5 6.75 7 7.25 7.5 7.75 8 8.25 8.5 8.75 9 9.25 9.5 9.75 10
    do
      python GenerateSubwordsBART.py -v_size $v_size \
                                     -dataset $data \
                                     -frac $frac \
                                     -csv_path ../FilterMedicalWords/TokenSplitDistribution-"$data"_BARTWithConsiderFlag.csv
    done
  done
done
