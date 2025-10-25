#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run collectstatic
python3.12 manage.py collectstatic --noinput --clear

# Run database migrations
python3.12 manage.py migrate