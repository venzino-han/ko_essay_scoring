default: build

help:
	@echo 'Management commands for kobert:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the  project project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t kobert 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name kobert -v `pwd`:/workspace/kobert kobert:latest /bin/bash

up: build run

rm: 
	@docker rm kobert

stop:
	@docker stop kobert

reset: stop rm