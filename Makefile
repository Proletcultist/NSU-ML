
PYTHON = python3
CALEPIN = calepin

REQUIREMENTS_FILE = requirements.txt
CALEPIN_CONFIG = calepin.toml

SRC_DIR = src
VENV_DIR = .venv
BUILD_DIR = build

all: first
first: ${BUILD_DIR}/1.pdf ${SRC_DIR}/1/data.csv

${VENV_DIR}: ${REQUIREMENTS_FILE}
	${PYTHON} -m venv ${VENV_DIR}
	${VENV_DIR}/bin/python3 -m pip install -r ${REQUIREMENTS_FILE}

${BUILD_DIR}/%.pdf: ${SRC_DIR}/%/main.typ ${CALEPIN_CONFIG} ${VENV_DIR}
	mkdir -p ${BUILD_DIR}
	${CALEPIN} compile --config ${CALEPIN_CONFIG} $<
	mv ${SRC_DIR}/$*/main.pdf $@

.PHONY: all first
