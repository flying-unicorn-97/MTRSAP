#!/bin/bash

echo "$HOSTNAME"
conda init bash &&
source activate icra24

# Set the number of times to run the Python script
# n=21
n=1

# Path to your Python script
python_script="./train_recognition.py"


# Loop to run the Python script n times
for ((i=0; i<$n; i++)); do
    echo "Running iteration $i"
    python "$python_script"  --model transformer --dataloader v1 --modality $i
done

echo "Script completed $n iterations."