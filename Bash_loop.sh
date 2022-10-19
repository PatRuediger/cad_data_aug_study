#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
FILES="/home/ruediger/augstudy/batch_scripts/*.sh"
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  sbatch "$f"
done
