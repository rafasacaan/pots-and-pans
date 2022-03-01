---
title: "Poetry for starters"
date: 2022-02-03T15:01:59-03:00
draft: True
---

Lately, while working at different data science projects, I have seen myself
repeated times in the situation of guiding others to build python
environments and on top, a specific set of dependencies. Suddenly, I realized
that for me this was never a trivial task: to create, mantain and handle on a
working python environment.

After searching the web, I have found that this task can be tackled by, first,
setting the foundations: a python environment, and second, build a house that
fits your needs: packages and dependencies.

## The foundations: python

An easy way to create a python environment is using the `pyenv` library,
but specifically the `pyenv-virtualenv` package will make our lives so much easier.
Lets install it in our local system (macosx):

```bash
brew install pyenv-virtualenv
```


Once installed, you can add different python versions to `pyenv` in order to make
them available as a foundation for our future python environments. For example,
let's add version 3.8.8:

```bash
pyenv install 3.8.8
```

Now, let's try and create our first environment.
```bash
pyenv virtualenv 3.8.8 my_first_env
```

As you can see, we have provided the python version required and a name for
our environment. If an error occurs and your env could not be created, try
adding the following two lines to your *.bash_profile*:
```
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Perfect! Now that you have created your first env, you can activate/deactivate it
by typing:

```bash
pyenv activate my_first_env
pyenv deactivate
```

You can check your created python envs by typing `pyenv virtualenvs` and check
the python version once inside with `pyenv version`. You can also remove an env
with `pyenv uninstall bye_bye_env`.

Great! Solid foundations have been stablished.

## The house: poetry for dependencies

After setting the foundations, we need to build our house based on different
modules that must fit each other. Here is where `poetry` shines and will become
your perfect dependency manager. Check their website and install it with the
following command:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Make sure it was correctly installed by typing `poetry --version`.

Now, we can got to our favorite directory and create a new project.
```bash
poetry new first_project
```
We can now add, search and remove python packages by typing:
```bash
poetry add <package>
poetry search <package>
poetry remove <package>
```

You can also tell poetry which python env to use by providing the path
```bash
poetry env use /usr/bin/python3
```

and enter the virtual env by running
```bash
poetry shell
```

and exit with
```bash
exit
```

To build from a poetry.lock file
```bash
poetry install
```

To update env
```bash
poetry update
```

To export to requirements.txt file:

```bash
poetry export -f requirements.txt -o requirements.txt
```

Check/delete poetry python envs:

```bash
poetry env list
poetry env remove /full/path/to/python
```

Check for poetry dependencies

```bash
poetry show
```


That's it!
