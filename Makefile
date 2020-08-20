.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = hrtf_model
PYTHON_INTERPRETER = python3



# Default model parameter

model_name = 'a' # needs to be set
exp_name= '' # needs to be set

azimuth = 12
snr = 0.2
freq_bands = 128
max_freq = 20000
normalize = False
time_window = 0.1  # time window in sec
elevations = 25
# filtering parameters
normalization_type = 'sum_1'
sigma_smoothing = 0
sigma_gauss_norm = 1
# use the mean subtracted map as the learned map
mean_subtracted_map = True
ear = 'contra'

# if the variable clean is set the model file is deleted and newly calculated

save_figs = False
save_type = 'svg'
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/data/generateData.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find ./models/ -type f -name "*.npy" -delete
	find ./models -type d -not -name models -delete

single_participant:
	test $(exp_name)
	test $(participant_number)

	# create model file
	$(PYTHON_INTERPRETER) src/models/single_participant_localization_exp/train_network.py --model_name='single_participant' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(steady_state) $(clean)
	$(PYTHON_INTERPRETER) src/models/single_participant_localization_exp/localize_sounds.py --model_name='single_participant' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(steady_state) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/single_participant/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='single_participant' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)

all_participants:
		test $(exp_name)
		test $(visualization_type)

		# create model file
		$(PYTHON_INTERPRETER) src/models/all_participants_localization_exp/localize_sound.py --model_name='all_participants' --exp_name=$(exp_name) --azimuth=$(azimuth) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
		# visualization
ifeq ($(visualization_type),regression)
	$(PYTHON_INTERPRETER) src/visualization/all_participants/visualize_regression_values.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='all_participants' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
else ifeq ($(visualization_type),sounds)
	$(PYTHON_INTERPRETER) src/visualization/all_participants/visualize_sound.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='all_participants' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
else ifeq ($(visualization_type),all)
	$(PYTHON_INTERPRETER) src/visualization/all_participants/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='all_participants' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
	$(PYTHON_INTERPRETER) src/visualization/all_participants/visualize_regression_values.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='all_participants' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
	$(PYTHON_INTERPRETER) src/visualization/all_participants/visualize_sound.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='all_participants' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
else
	$(PYTHON_INTERPRETER) src/visualization/all_participants/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='all_participants' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
endif


different_learned_maps:
		test $(exp_name)
		test $(visualization_type)

		# create model file
		$(PYTHON_INTERPRETER) src/models/different_learned_maps/localize_sound.py --model_name='different_learned_maps' --exp_name=$(exp_name) --azimuth=$(azimuth) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
		# visualization
ifeq ($(visualization_type),regression)
	$(PYTHON_INTERPRETER) src/visualization/different_learned_maps/visualize_regression_values.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='different_learned_maps' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
else ifeq ($(visualization_type),all)
	$(PYTHON_INTERPRETER) src/visualization/different_learned_maps/visualize_regression_values.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='different_learned_maps' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
	$(PYTHON_INTERPRETER) src/visualization/different_learned_maps/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='different_learned_maps' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
else
	$(PYTHON_INTERPRETER) src/visualization/different_learned_maps/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='different_learned_maps' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
endif


hrtf_comparison:
	test $(exp_name)
	test $(participant_number)
	test $(visualization_type)

	# create model file
	$(PYTHON_INTERPRETER) src/models/hrtf_comparison_exp/comparison_single_participant.py --model_name='hrtf_comparison' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/hrtf_comparison/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='hrtf_comparison' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)

hrtf_creation:
	test $(exp_name)
	test $(participant_number)
	$(PYTHON_INTERPRETER) src/models/hrtf_creation/comparison_single_participant.py --model_name='hrtf_creation' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/hrtf_creation/visualize_single_hrtf.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='hrtf_creation' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)

create_elevation_spectra_maps:
		test $(exp_name)
		# create model file
		$(PYTHON_INTERPRETER) src/models/all_elevation_spectra_maps/create_maps.py --model_name='elevation_spectra_maps' --exp_name=$(exp_name) --azimuth=$(azimuth) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --participant_numbers=$(participant_numbers) $(clean)
		# visualization
		$(PYTHON_INTERPRETER) src/visualization/all_elevation_spectra_maps/visualize_raw_maps.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='elevation_spectra_maps' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --participant_numbers=$(participant_numbers)


map_learning:
	test $(exp_name)
	test $(n_trials)
	test $(visualization_type)

	# create model file
ifeq ($(visualization_type),all_maps)
	$(PYTHON_INTERPRETER) src/models/map_learning_exp/map_learning_all_maps.py --model_name='map_learning_all_maps' --exp_name=$(exp_name) --azimuth=$(azimuth) --n_trials=$(n_trials) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/map_learning/visualize_all_maps.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='map_learning_all_maps' --exp_name=$(exp_name) --azimuth=$(azimuth) --n_trials=$(n_trials) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
else ifeq ($(visualization_type),all)
	$(PYTHON_INTERPRETER) src/models/map_learning_exp/map_learning_all_maps.py --model_name='map_learning_all_maps' --exp_name=$(exp_name) --azimuth=$(azimuth) --n_trials=$(n_trials) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/map_learning/visualize_all_maps.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='map_learning_all_maps' --exp_name=$(exp_name) --azimuth=$(azimuth) --n_trials=$(n_trials) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
	$(PYTHON_INTERPRETER) src/models/map_learning_exp/map_learning.py --model_name='map_learning' --exp_name=$(exp_name) --azimuth=$(azimuth) --n_trials=$(n_trials) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/map_learning/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='map_learning' --exp_name=$(exp_name) --azimuth=$(azimuth) --n_trials=$(n_trials) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
else
	$(PYTHON_INTERPRETER) src/models/map_learning_exp/map_learning.py --model_name='map_learning' --exp_name=$(exp_name) --azimuth=$(azimuth) --n_trials=$(n_trials) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/map_learning/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='map_learning' --exp_name=$(exp_name) --azimuth=$(azimuth) --n_trials=$(n_trials) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)
endif



parameter_sweep:
	test $(exp_name)

	# create model file
	$(PYTHON_INTERPRETER) src/models/parameter_sweep_exp/parameter_sweep.py --model_name='parameter_sweep' --exp_name=$(exp_name) --azimuth=$(azimuth) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type)  $(clean)
	$(PYTHON_INTERPRETER) src/visualization/parameter_sweep/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='parameter_sweep' --exp_name=$(exp_name) --azimuth=$(azimuth) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type)


snr_exp:
	test $(exp_name)

	# create model file
	$(PYTHON_INTERPRETER) src/models/snr_exp/snr.py --model_name='snr_experiment' --exp_name=$(exp_name) --azimuth=$(azimuth)  --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/snr/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='snr_experiment' --exp_name=$(exp_name) --azimuth=$(azimuth) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)




# run all models and visualizations
all: 	single_participant all_participants different_learned_maps hrtf_comparison hrtf_creation parameter_sweep map_learning snr_exp 







#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
