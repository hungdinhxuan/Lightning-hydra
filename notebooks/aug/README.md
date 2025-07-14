# AUTOTUNE MODULE


# Most stable (sequential processing)
./generate.sh -q -c -s 42

# Threaded processing (faster but still stable)
./generate.sh -c -w 2 -s 42

# Full sequential for debugging
./generate.sh -q -s 42