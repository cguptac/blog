#!/bin/bash

for j in `seq 1 2` ; do  
    for i in `seq -f %02g 1 20` ; do
        if [ "$j" -lt "2" ]; then
            append=train
        else
            append=test
        fi
        fname=task_${i}_${j}_${append}
        babi-tasks ${i} 1000 > ${fname}
        printf "                      \r%s\r" "${fname}"
    done
done

tar -cvf tasks.tar task_*_*_*
rm task_*_*_*
