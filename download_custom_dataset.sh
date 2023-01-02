# init folder
mkdir -p data

# download custom ShapeNet multi-view dataset for training
cd data
wget ftp://213.144.153.67/ObjectReconstructor/small_train_test_split.tar.gz
tar zxvf small_train_test_split.tar.gz
rm small_train_test_split.tar.gz
cd ..