.PHONY: all clean

all: venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv -p python3.6 venv
	. venv/bin/activate && pip install -U jupyter
	. venv/bin/activate && pip install -U matplotlib
	. venv/bin/activate && pip install -U RISE
	. venv/bin/activate && jupyter-nbextension install rise --py --sys-prefix
	. venv/bin/activate && jupyter-nbextension enable rise --py --sys-prefix

clean:
	rm -fr venv
	find -iname "*.pyc" -delete

