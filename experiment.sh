#!/bin/sh

./scripts/setup.sh
./scripts/evaluate_pythia_410m.sh
./scripts/wasserstein_distance.sh
./scripts/group_neurons.sh
