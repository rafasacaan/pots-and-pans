---
title: "virutalenv + pip"
date: 2022-02-03T15:01:59-03:00
draft: True
---

This is a cheat list for using **virtualenv** to create python environments, and **pip** as an install manager.

## virtualenv.

Create a virtualenv:
> python3 -m virtualenv my_env

Activate the env:
> source my_env/bin/activate

Deactivate:
> deactivate


Remove an env:
> rm -rf my_env


## pip
Generate file with packages:
> my_env/bon/python -m pip freeze > requirements.txt

Install packages in env from file:
> my_env/bon/python -m pip install -r requirements.txt

List packages:
> pip list

Install new packages:
> python3 -m pip install jupyterlab scikit-learn pandas numpy spicy matplotlib

### End!