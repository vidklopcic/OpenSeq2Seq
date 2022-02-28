# install Boost package
sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev swig
git clone --depth 1 https://github.com/vidklopcic/OpenSeq2Seq -b ctc-decoders
mv OpenSeq2Seq/decoders .
rm -rf OpenSeq2Seq
cd decoders
./setup.sh
cd ..
