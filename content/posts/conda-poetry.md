---
title: "conda + poetry"
date: 2022-02-03T15:01:59-03:00
draft: True
---

Whenever starting a new project, it is good to get started as nice and easy as possible. The fundamental bricks for this task is to generate an environment and a basic tree of files and directories. And of course, the necessary tools to keep things working from then on.

What else? We can treat our projects as packages right away, and get an awesome package manager such as poetry.

## Conda, the environment-maker-manager.

Let´s create the most basic python environment, where the most important thing is to have a defined working python version. To do this, we can create a conda lite-weight environment from the following *.yaml* file:

```bash
name: my_env_name

channels:
  - default
  - conda-forge

dependencies:
  - python=3.8

```

Please notice that a `name` and `python version` are defined inside the file. Now, we can create our very basic env.

```bash
conda env create -f environment.yaml
```

That´s it with `conda`!

## Poetry, the hard-working-friend.

First, we need to install poetry using:
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

After installing `poetry`, we can run the following command on the terminal.

```bash
poetry new my_project
```

Several files will be created to get started, among these, a `pyproject.toml` file which is in charge to manage our dependencies.

Now, we can do the following:
- We can install our dependencies, if added using `poetry install`. 
-  Install packages for production with `poetry add <a-name>`
-  Install packages for development with `poetry add <a-name> --dev`
- Remove packages with `poetry remove <a-name>`
-  Show packages with `poetry show`

Also, if we need to run an existing project, we can do `poetry init` to create the *.toml* file.

## Sharing is caring

Now, we can export our conda env, which was managed by poetry.

```bash
conda env export > environment.yml
```
With this file, we can recreate our environment with conda and run `poetry install` to get things up and running.

Easy as a-b-c.

## Note!

Something that helped a lot, was to restrict the space of python versions compatibility. I did this in the **pyproject.toml** file:
```
python = ">=3.8,<3.9"
```