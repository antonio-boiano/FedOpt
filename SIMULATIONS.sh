#!/bin/bash

# Working directory (current directory)
WORKSPACE_DIR=$(pwd)

# Function to terminate all Python processes and free the port
cleanup() {
    echo "Cleaning up background jobs..."
    kill $(jobs -p) 2>/dev/null
    echo "Freeing port $1 if occupied..."
    lsof -ti :$1 | xargs kill -9 2>/dev/null
    echo "Cleanup completed."
}

# Function to run a simulation and save the output in the log file
run_simulation() {
    local mode=$1
    local protocol=$2
    local fl_method=$3
    local port=$4
    local client_number=$5
    local alpha=$6
    local mu=$7
    local variant=$8
    local file_name=$9


    local log_file="$LOG_DIR/${mode}_${client_number}.log"
    
    echo "Running: mode=$mode, fl_method=$fl_method, port=$port"
    python3 "$WORKSPACE_DIR/FedOpt/src/run.py" --mode "$mode" --rounds 5 --protocol "$protocol" --port "$port"  --fl_method "$fl_method"  --alpha "$alpha" --accuracy_file "$file_name" &> "$log_file" &
    echo "Log saved to $log_file"
}

PORT="8082"
PROTOCOL="grpc"

# PORT=1883
# PROTOCOL="mqtt"

NUM_CLIENTS=5

ALGORITHMs=("Unweighted")
MUs=(0.1)
ALPHA=(10)
VARIANTS=("hinge" "polynomial")



# # Porta usata per il server e i client (modifica qui se necessario)
# PORT="50051"
# PROTOCOL="grpc"
# NUM_CLIENTS=3


# # Algorithms to execute
# ALGORITHMs=("ASOFed")
# # Parameters for FedAsync
# MUs=(0.1)
# ALPHA=(10)
# VARIANTS=("hinge" "polynomial")

# Run cleanup before starting new simulations
cleanup "$PORT"

# # If using MQTT protocol, start the Mosquitto server
if [ "$PROTOCOL" == "mqtt" ]; then
    mosquitto -p "$PORT" -d
fi

# Ensure cleanup runs when the script exits
trap 'cleanup "$PORT"' EXIT

# Loop through the algorithms
# Create a directory for logs (if it doesn't exist)
for alpha in "${ALPHA[@]}"; do
    for ALGORITHM in "${ALGORITHMs[@]}"; do
        if [ "$ALGORITHM" == "FedAsync" ]; then
            for variant in "${VARIANTS[@]}"; do
                for MU_0 in "${MUs[@]}"; do
                    echo "Starting simulations for algorithm: $ALGORITHM"
                    LOG_DIR="$WORKSPACE_DIR/logs/$PROTOCOL/$ALGORITHM/$variant/alpha${alpha}_10clients_2"
                    rm -rf "$LOG_DIR"
                    mkdir -p "$LOG_DIR"

                    # Start the server
                    run_simulation "server" "$PROTOCOL" "$ALGORITHM" "$PORT" "0" "$alpha" "$MU_0" "$variant" "$LOG_DIR/accuracy_alpha${alpha}.png"
                    sleep 5 # Wait for the server to start

                    # Start client 1 to NUM_CLIENTS
                    for i in $(seq 1 $NUM_CLIENTS); do
                        run_simulation "client" "$PROTOCOL" "$ALGORITHM" "$PORT" "$i" "$alpha" "$MU_0" "$variant" "$WORKSPACE_DIR/logs/$PROTOCOL/$ALGORITHM/$variant/accuracy_alpha${alpha}.png"
                    done   
                    # Wait for all background processes to finish
                    wait
                done
            done
        else
            for MU_0 in "${MUs[@]}"; do
                echo "Starting simulations for algorithm: $ALGORITHM"
                LOG_DIR="$WORKSPACE_DIR/logs/$PROTOCOL/$ALGORITHM/alpha${alpha}_10clients_2"
                rm -rf "$LOG_DIR"
                mkdir -p "$LOG_DIR"

                # Start the server
                run_simulation "server" "$PROTOCOL" "$ALGORITHM" "$PORT" "0" "$alpha" "$MU_0" "" "$LOG_DIR/accuracy_alpha${alpha}_10clients_2.png"
                sleep 5 # Wait for the server to start

                # Start client 1 to NUM_CLIENTS
                for i in $(seq 1 $NUM_CLIENTS); do
                    run_simulation "client" "$PROTOCOL" "$ALGORITHM" "$PORT" "$i" "$alpha" "$MU_0" "" "$WORKSPACE_DIR/logs/$PROTOCOL/$ALGORITHM/accuracy_alpha${alpha}_10clients_2.png"
                done   
                # Wait for all background processes to finish
                wait
            done
        fi
    done
done

echo "All simulations have been completed!"
