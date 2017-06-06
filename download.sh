# Use this script to download the entire LibriSpeech dataset

#!/bin/bash

base="http://www.openslr.org/resources/12/"



for s in 'dev-clean' 'test-clean' 'train-clean-100'
do
    linkname="${base}/${s}.tar.gz"
    echo $linkname
    wget -c $linkname
done

for s in 'dev-clean' 'test-clean' 'train-clean-100'
do
    tar -xzvf $s.tar.gz
done

