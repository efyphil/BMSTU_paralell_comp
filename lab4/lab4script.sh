#/bin/bash
#PBS -k o
#PBS -j oe
#$PBS_O_WORKDIR/a.out $PBS_O_WORKDIR/out.txt 800 537 1
PBS_O_WORKDIR=/home/student21m01/labs/lab4

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

echo "===== First mygroup ====="

for i in {1..3}
do
echo 
        python3 $PBS_O_WORKDIR/test.py --gpu --workdir $PBS_O_WORKDIR \
          --gen-length 1000 --epoch-number 19
        python3 $PBS_O_WORKDIR/test.py --rus --gpu --workdir $PBS_O_WORKDIR \
#         --gen-length 1000 --epoch-number 40 --bigger-model
done

echo "===== Second mygroup ====="

for i in {1..10}
do
        python3 $PBS_O_WORKDIR/test.py --gpu --cpu --workdir $PBS_O_WORKDIR \
          --gen-length 1000 --epoch-number 19
         python3 $PBS_O_WORKDIR/test.py --cpu --workdir $PBS_O_WORKDIR \
        #   --gen-length 1000 --epoch-number 19
done

echo "===== Third mygroup ====="
for i in 88 128 168
do
echo
        python3 $PBS_O_WORKDIR/test.py --gpu --workdir $PBS_O_WORKDIR \
#         --gen-length $i --epoch-number 19
done

echo "===== Fourth mygroup ====="

for i in 1 20 40 50
do
echo
        python3 $PBS_O_WORKDIR/test.py --rus --gpu --workdir $PBS_O_WORKDIR \
#         --gen-length 200 --epoch-number $i --bigger-model
done

echo "===== Fifth mygroup ====="
user_seq="Architecture the art of building in which human requirements and"
user_seq="$user_seq construction materials are related so as to furnish practical"
user_seq="$user_seq use as well as an aesthetic solution"

python3 $PBS_O_WORKDIR/test.py --gpu --workdir $PBS_O_WORKDIR \
#         --gen-length 200 --start-sequence "$user_seq"

#echo "Lab4 script execution is over. Please, check results" 
