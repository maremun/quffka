#!/bin/bash
D="datasets"
# CCPP Powerplant
DF="Powerplant"
mkdir -p "$D/$DF";
wget -nc -O "$D/$DF/data.zip" \
    https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip
unzip -u -d "$D/$DF/" "$D/$DF/data.zip"
# LETTER
DF="LETTER"
mkdir -p "$D/$DF";
wget -nc -O "$D/$DF/data" \
    https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data
# USPS
DF="USPS"
mkdir -p "$D/$DF";
wget -nc -O "$D/$DF/data.bz2" \
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
wget -nc -O "$D/$DF/data_test.bz2" \
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
bzip2 -dk "$D/$DF/data.bz2" "$D/$DF/data_test.bz2"
# CIFAR100
DF="CIFAR100"
mkdir -p "$D/$DF";
wget -nc -O "$D/$DF/data.tar.gz" https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -C "$D/$DF/" -xf "$D/$DF/data.tar.gz"
# LEUKEMIA
DF="LEUKEMIA"
mkdir -p "$D/$DF";
wget -nc -O "$D/$DF/leu.bz2" \
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.bz2
wget -nc -O "$D/$DF/leu.t.bz2" \
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.bz2
bzip2 -dk "$D/$DF/leu.bz2" "$D/$DF/leu.t.bz2"
