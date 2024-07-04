#!/bin/bash

ExitIfFailed()
{
    if [ $1 != 0 ]; then
        echo "!!!!! Build step ($2) failed! Exiting !!!!!"
        exit 1
    fi
}

cd $BUILD_REPOSITORY_LOCALPATH

rm -rf venv_build
python3 -m virtualenv venv_build
source venv_build/bin/activate
pip install flake8

flake8 --exclude=.git,__pycache__,build,venv_build,.venv,code/model/twinbert --max-line-length=120 -j 1
ExitIfFailed $? "Linting failed"
echo "Linting successful"

rm -rf venv_build