#!/bin/bash

# Set the CUDA version-related paths
CUDA_HOME=/usr/local/cuda
export CUDA_HOME

# Add CUDA binaries to PATH
export PATH=$CUDA_HOME/bin:$PATH

# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Write the inputs to the .env file and CUDA settings
cat <<EOL > .env
BIDSDATAPATH=$BIDSDATAPATH
EOL
echo ".env file has been created/updated successfully with BIDSDATAPATH!"

# CUDA PATHS for CUPY
cuda_config="
# Set the CUDA version-related paths
CUDA_HOME=/usr/local/cuda
export CUDA_HOME

# Add CUDA binaries to PATH
export PATH=\$CUDA_HOME/bin:\$PATH

# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
"

# Check if the CUDA configuration is already in the .bashrc file
if ! grep -q "CUDA_HOME=/usr/local/cuda" ~/.bashrc; then
    # If not, append it to the .bashrc
    echo "$cuda_config" >> ~/.bashrc
    echo "CUDA paths have been added to your ~/.bashrc"
else
    echo "CUDA paths are already set in your ~/.bashrc"
fi

# Source the updated ~/.bashrc to apply changes immediately
source ~/.bashrc
echo "Run the following to apply changes to CUDA paths"
echo "source ~/.bashrc "


