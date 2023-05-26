#!/bin/bash

# Define a function to convert a Jupyter notebook to a Python script
jupyter_to_python() {
    jupyter nbconvert --to python "$1"
}

# Define a function to run a Python script
run_python_script() {
    python3 "$1" -n 12
}

# Define a function to delete a file
delete_file() {
    rm -f "$1"
}

for notebook in "$@"
do
    # Convert the current notebook to Python and save it in the same folder as the original notebook
    py_script="$(dirname "${notebook}")/$(basename "${notebook}" .ipynb).py"  
    jupyter_to_python "${notebook}" --output="${py_script}"

    # Run the resulting Python script      
    run_python_script "${py_script}"
    
    # Delete the resulting Python script          
    delete_file "${py_script}"
done
