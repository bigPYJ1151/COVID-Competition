
filename=$1
fold_num=$2

if [[ $filename = "" ]];
then
    echo "Arg1 filename missing!"
    exit;
fi;

if [[ $fold_num = "" ]];
then
    echo "Arg2 fold_num missing!"
    exit;
fi;

for ((i=0;i<$fold_num;i++));
do
    python $filename -f $i
done;