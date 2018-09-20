C++ Extension Talk
==================


These are files accompanying a talk given in Stockholm on September 20th, 2018.
The folders and code are set up to make things very simple. I'm using a weird
setup with symlinks and things to even avoid using a virtual environment. You
should basically just be able to go into each folder and run `make` with
different make rules (see the `Makefile` in each directory) and everything
should just work.

If you are running debian 9 (and probably Ubuntu), the following requirements
should be enough:

```
    $ apt install vim make python3-dev python3-setuptools \
        python3-sklearn python3-matplotlib imagemagick gdb
```
