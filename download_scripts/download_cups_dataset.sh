# init folder
mkdir -p data

# download custom ShapeNet multi-view dataset for training
cd data
wget ftp://213.144.153.67/ObjectReconstructor/single_category_cups_train_test_split.tar.gz
tar zxvf single_category_cups_train_test_split.tar.gz
rm single_category_cups_train_test_split.tar.gz
cd ..