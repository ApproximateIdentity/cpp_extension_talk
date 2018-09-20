export PYTHONPATH=$(PWD)

build:
	python3 setup.py build --build-lib $(PWD)

profile: build
	python3 profile.py

profile-pythreads: build
	python3 profile_pythreads.py

clean:
	rm -rf __pycache__ *.png *.pyc build/ *.so
