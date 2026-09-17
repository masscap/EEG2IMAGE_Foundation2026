# 2. Build the caches under CACHE_DIR (~13 GB): the preprocessed windows,
#    plus one frozen DINOv2-giant embedding per unique stimulus (~100 MB
#    for THINGS-EEG2, content-keyed and shared with the other image
#    tasks). The only --prepare of the four tracks that needs a GPU.
#    ~15 min for eegnet's cache, ~45 min for reve's, and ~10 min to embed
#    the 16740 stimuli, spread over 10 and 128 SLURM jobs respectively.
neuralbench eeg image --prepare

# 3. Sanity check before you queue anything: 2 epochs, a data subset, one
#    seed, always in-process, so progress lands in your terminal. ~2 min
#    on one V100 with the cache warm. Name the model you actually plan to
#    run -- a bare --debug takes the config default, which is EEGNet.
neuralbench eeg image -m eegnet --debug

# 4. Same check for the foundation model. REVE's weights are gated on the
#    HuggingFace Hub, so this needs an account and an accepted licence;
#    it is the cheapest place to discover that, because a queued run
#    reports the failure into a job log instead of your terminal.
neuralbench eeg image -m reve --debug

# 5. Full baseline -- task-specific model (EEGNet). ~2.5 h per seed, and
#    the default grid is three seeds (concurrent on SLURM).
neuralbench eeg image -m eegnet

# 6. Full baseline -- foundation model (REVE), fine-tuned end to end.
#    ~5.5 h per seed. ~69M parameters against EEGNet's ~1.5k, all of them
#    trainable here, so this one wants a datacentre GPU rather than a
#    laptop; it also preprocesses at 200 Hz against the 120 Hz default,
#    warming a second cache.
neuralbench eeg image -m reve