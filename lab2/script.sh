
#!/bin/bash
#PBS -k o
#PBS -j oe


#3 5 7 9 17 21

for stride in 1
do
  for i in {1..10}
  do
    echo "=============="
    echo "Stride $stride"
    echo "=============="
    $PBS_O_WORKDIR/cpu_21x21 $PBS_O_WORKDIR/out.txt 960 536 $stride
    echo "============="
    echo "gpu++++++++++"
    $PBS_O_WORKDIR/gpu_21x21 $PBS_O_WORKDIR/out.txt 960 536 $stride
  done
done


