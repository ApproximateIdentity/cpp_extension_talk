export PYTHONPATH=$(PWD)

build:
	python3 setup.py build --build-lib $(PWD)

plot-decision-boundary: build
	python3 plot_decision_boundary.py
	display points.png &
	display decision_boundary.png &

test: build
	python3 -m pytest -v

profile: build
	python3 profile.py
	display profile.png &

clean:
	rm -rf __pycache__ *.png *.pyc build/ *.so
