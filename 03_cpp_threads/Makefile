export PYTHONPATH=$(PWD)

build:
	python3 setup.py build --build-lib $(PWD)

test: build
	python3 -m pytest -v

profile: build
	python3 profile.py
	display profile.png &

profile-pythreads: build
	python3 profile_pythreads.py
	display profile_pythreads.png &

clean:
	rm -rf __pycache__ *.png *.pyc build/ *.so
