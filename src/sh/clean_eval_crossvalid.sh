
model_tag=$1
fold_num=$2

if [[ $model_tag = "" ]];
then
    echo "Arg1 model_tag missing!"
    exit;
fi;

if [[ $fold_num = "" ]];
then
    echo "Arg2 fold_num missing!"
    exit;
fi;

for ((i=0;i<$fold_num;i++));
do
    rm -f -r ../record/"$model_tag"_fold"$i".pth/data/*
    echo ../record/"$model_tag"_fold"$i".pth/data/,cleaned.
done;