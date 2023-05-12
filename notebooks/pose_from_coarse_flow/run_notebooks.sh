#!/bin/bash

# Define a function to convert a Jupyter notebook to a Python script
jupyter_to_python() {
    jupyter nbconvert --to python "$1"
}

# Define a function to run a Python script
run_python_script() {
    ipython "$1"
}

# Define a function to delete a file
delete_file() {
    rm -f "$1"
}

for notebook in "$@"
do
    # Convert the current notebook to Python
    jupyter_to_python "${notebook}"

    # Run the resulting Python script and capture any output/errors in variables
    py_script="${notebook%.ipynb}.py"  # Get the name of the resulting Python script
    run_python_script "${py_script}"
    
    # Delete the resulting Python script
    delete_file "${py_script}"
done


