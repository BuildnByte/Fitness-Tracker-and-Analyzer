#!/bin/bash
set -e

# Upgrade pip and install dependencies
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -r requirements.txt

# Run collectstatic
python3.12 manage.py collectstatic --noinput --clear

# Run database migrations
python3.12 manage.py migrate