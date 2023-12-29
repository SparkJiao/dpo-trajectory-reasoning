name=$1
steps=$@
echo $name
#echo $steps
#x=$#
# enumerate all parameters since $name
for ((i=2;i<=$#;i++)); do
    echo ${!i}
done