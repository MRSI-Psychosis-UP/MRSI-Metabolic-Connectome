#!/bin/bash

echo "Select an option:"
echo "1) Build a new environment"
echo "2) Enter an existing environment"
echo "3) Exit the current environment"
read -p "Enter your choice (1/2/3): " choice

case $choice in
    1)
        echo "Checking if the environment 'neurocombat_env' already exists..."
        
        # Check if the environment exists by name
        if conda env list | grep -q 'neurocombat_env'; then
            echo "Environment 'neurocombat_env' already exists. Removing it..."
            conda env remove -n neurocombat_env
        else
            echo "No environment with name 'neurocombat_env' found."
        fi

        # Prompt for path if the environment doesn't have a name
        read -p "Do you want to associate an existing environment by path? (y/n): " associate_by_path
        if [[ $associate_by_path == "y" || $associate_by_path == "Y" ]]; then
            read -p "Enter the full path to the environment: " env_path
            if [ -d "$env_path" ]; then
                echo "Activating the environment using the provided path..."
                conda activate "$env_path"
                
                # Export environment from the path
                echo "Exporting environment from path..."
                conda env export --prefix "$env_path" > temp_environment.yaml
                
                # Remove the environment by path
                echo "Removing environment by path..."
                conda env remove --prefix "$env_path"
                
                # Recreate the environment with the correct name
                echo "Creating the environment 'neurocombat_env'..."
                conda env create --name neurocombat_env --file temp_environment.yaml
                echo "Environment created with the name 'neurocombat_env'."
                
                # Clean up the temporary environment file
                rm temp_environment.yaml
            else
                echo "Invalid path. No environment found at '$env_path'. Exiting."
                exit 1
            fi
        else
            # Create a new environment from a YAML file
            echo "Creating a new environment..."
            if conda env create -n neurocombat_env -f environment.yaml; then
                echo "Environment 'neurocombat_env' created."
            else
                echo "Failed to create the environment."
                exit 1
            fi
        fi
        ;;
    2)
        echo "Activating the 'neurocombat_env' environment..."
        conda activate neurocombat_env
        if [ $? -eq 0 ]; then
            echo "Environment 'neurocombat_env' activated."
        else
            echo "Failed to activate the environment. It may not exist."
            exit 1
        fi
        ;;
    3)
        echo "Deactivating the current environment..."
        conda deactivate
        echo "Environment deactivated."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
