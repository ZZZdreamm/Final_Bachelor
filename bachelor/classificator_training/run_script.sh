#!/bin/bash

PYTHON_SCRIPT="classificator_training.main"

PARAMETER_SETS=(
    # "1 1 0 0 1 64 hardmining 1 0.000005 0.2"
    # "1 1 0 0 1 64 hardmining 2 0.00001 0.2"
    # "1 1 1 1 1 64 hardmining 1 0.0001 0.2" #
    # "1 1 0 0 1 64 hardmining 0 0.00001 0.2" 
    # "1 1 0 0 1 64 hardmining 0 0.000005 0.2"
    # "1 1 0 0 1 64 hardmining 1 0.00001 0.2" 
    # "1 1 0 0 1 64 hardmining 1 0.000005 0.2"
    # "1 1 0 0 1 64 hardmining 0 0.00001 0.5" 
    # "1 1 0 0 1 64 hardmining 0 0.00001 0.1" 
    # "1 1 1 1 1 64 hardmining 0 0.00001"
    # "1 1 0 0 1 64 standard 0 0.00001" 
    # "1 1 0 0 1 64 curriculum 0 0.00001" # Be
    # "1 1 1 1 0 64 hardmining 1 0.000005 0.2"
    # "1 1 1 0 0 64 hardmining 1 0.000005 0.2"
    # "1 1 0 0 0 64 hardmining 1 0.000005 0.2"
    # "1 1 1 1 0 64 hardmining 1 0.000005 0.2"
    # "1 0 0 0 0 64 hardmining 1 0.000005 0.2"
    # "0 1 0 0 0 64 hardmining 1 0.000005 0.2"
    # "0 0 1 0 0 64 hardmining 1 0.000005 0.2"
    # "0 1 1 0 1 64 hardmining 1 0.000005 0.2"

    
    # "1 0 0 0 0 32 hardmining 2 0.000001 0.2"   

    # "1 1 1 1 1 32 hardmining 2 0.00001 0.2"    
    # "1 0 1 0 1 64 hardmining 2 0.000005 0.5"   
    # "1 0 0 0 0 64 hardmining 1 0.000005 0.2 0" 
    # "1 0 0 0 1 64 hardmining 1 0.000005 0.2 0" 
    # "1 0 0 0 1 64 hardmining 2 0.000005 0.2 0" 
    # "1 0 0 0 0 64 hardmining 2 0.000001 0.2 0"   
    # "1 0 0 0 1 64 hardmining 2 0.000001 0.2 0"   
    # "1 1 0 0 1 64 hardmining 2 0.000005 0.2 0"   





    # "1 0 0 0 0 64 hardmining 2 0.000001 0.2 0"   
    # "1 0 0 0 0 64 hardmining 2 0.000001 0.2 0 32.0"  
    # "1 1 1 1 1 64 hardmining 2 0.000001 0.2 0 32.0"   
    # "1 1 1 1 1 64 hardmining 2 0.000005 0.2 0 32.0"   
    # "1 0 0 0 0 64 hardmining 2 0.000001 0.2 0 64.0"  
    # "1 0 0 0 0 64 hardmining 2 0.000005 0.2 0 64.0" # NAJLEPSZYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    # "1 0 0 0 0 64 hardmining 2 0.000005 0.4 0 64.0" BEST 2
    # "1 0 0 0 0 64 hardmining 2 0.000001 0.2 0 48.0"  










    # "1 0 0 0 0 64 hardmining 2 0.000005 0.2 0 32.0"   # BEST 3
    # "1 0 0 0 0 64 hardmining 2 0.000005 0.4 0 32.0"  # BEST 4
    # "1 0 0 0 0 64 hardmining 3 0.000005 0.2 0 32.0"   
    # "1 0 0 0 0 64 hardmining 3 0.000005 0.4 0 32.0" 

    # "1 0 0 0 1 64 hardmining 2 0.000005 0.2 0 32.0" 
    # "1 0 0 0 1 64 hardmining 2 0.000005 0.4 0 32.0"
    # "1 0 0 0 1 64 hardmining 3 0.000005 0.2 0 32.0" 
    # "1 0 0 0 1 64 hardmining 3 0.000005 0.4 0 32.0"


    # Duzy head z duzym marginesem radzi sobie dobrze 
    # "1 0 0 0 0 64 hardmining 3 0.000001 0.8 0 64.0"  # BEST 5

    # "1 0 0 0 0 64 hardmining 3 0.000001 1.2 0 64.0"
    # "1 0 0 0 0 64 hardmining 3 0.0000004 1.2 0 64.0"
    # "1 0 0 0 0 64 hardmining 3 0.0000004 0.8 0 64.0"
    # "1 1 1 1 1 64 hardmining 3 0.000001 0.8 0 64.0" 
    # "1 1 1 1 1 64 hardmining 3 0.0000002 0.8 0 64.0" 
    # "1 1 1 1 1 64 hardmining 3 0.0000002 1.2 0 64.0" 
    # "1 0 0 0 0 64 hardmining 3 0.0000004 1.6 0 64.0" # BEST 6
    # "1 0 0 0 0 64 hardmining 3 0.000001 0.4 0 64.0" 
    # "1 0 0 0 1 64 hardmining 3 0.0000004 1.2 0 64.0"
    # "1 1 0 0 1 64 hardmining 3 0.0000004 1.2 0 64.0"

    # "1 0 0 0 0 64 hardmining 2 0.0000004 0.2 0 64.0"
    # "1 0 0 0 0 64 hardmining 2 0.0000004 0.4 0 64.0"
    # "1 0 0 0 0 64 hardmining 2 0.0000004 0.8 0 64.0"

    # "1 0 0 0 0 64 hardmining 2 0.0000002 0.8 0 64.0" # BEST 7
    # "1 0 0 0 0 64 hardmining 2 0.0000002 1.2 0 64.0"

    # "1 0 0 0 0 64 hardmining 2 0.0001 1.2 0 64.0"

    # "1 0 0 0 0 128 hardmining 3 0.0000004 1.2 0 64.0"
    # "0 0 0 0 1 0 0 0 0 64 hardmining 3 0.000002 1.2 64.0"

    # "1 0 0 0 0 0 0 0 0 64 hardmining 2 0.000001 1.2 64.0"


    # "0 0 0 0 1 0 0 0 0 64 hardmining 2 0.000002 0.4 32.0"

    # "0 0 0 0 0 1 0 0 0 64 hardmining 3 0.00002 0.4 32.0"
    # "0 0 0 0 0 1 0 0 0 64 hardmining 2 0.000002 0.4 32.0"

    # "0 0 0 0 0 0 1 0 0 64 hardmining 3 0.000002 0.4 32.0"
    # "0 0 0 0 0 0 1 0 0 64 hardmining 2 0.00002 0.4 32.0"

    # "0 0 0 0 0 0 0 1 0 64 hardmining 3 0.000002 0.4 32.0"
    # "0 0 0 0 0 0 0 1 0 64 hardmining 2 0.000002 0.4 32.0"

    # "0 0 0 0 0 0 1 1 0 64 hardmining 2 0.00002 0.4 32.0"

    # "0 0 0 0 1 0 0 1 0 64 hardmining 2 0.000002 0.4 32.0"

    # "0 0 0 0 0 0 1 1 1 64 hardmining 2 0.00002 0.4 32.0"


    # "0 0 0 0 0 0 0 1 1 64 hardmining 2 0.00002 0.4 32.0"
    # "0 1 0 0 0 0 0 0 0 64 hardmining 3 0.000002 1.2 64.0"
    # "0 0 1 0 0 0 0 0 0 64 hardmining 3 0.000002 1.2 64.0"
    # "0 0 0 1 0 0 0 0 0 64 hardmining 3 0.000002 1.2 64.0"
    "1 0 1 0 0 0 0 0 1 64 hardmining 2 0.000002 1.2 64.0"
    "1 0 1 0 0 0 0 0 0 64 hardmining 2 0.000002 1.2 64.0"
    "1 0 0 0 0 0 0 0 0 64 standard 2 0.000002 1.2 64.0"
    "1 0 0 0 0 0 0 1 0 64 hardmining 2 0.000002 1.2 64.0"
    # "1 0 0 0 0 64 hardmining 3 0.00005 1.2 0 64.0"
    # "1 0 0 0 0 64 hardmining 3 0.000003 1.2 0 64.0"
    # "1 0 0 0 0 64 hardmining 3 0.0000004 1.6 0 64.0"
    # "1 0 0 0 0 64 hardmining 2 0.0000002 0.8 0 64.0"

    # "1 0 0 0 0 64 hardmining 3 0.000001 0.8 0 64.0"
    # "1 0 0 0 0 64 hardmining 2 0.00001 0.2 0 64.0" 
    # "1 0 0 0 0 64 hardmining 2 0.00001 0.2 0 64.0" 
    # "1 0 0 0 0 64 hardmining 2 0.000005 0.2 0 32.0"
    # "1 0 0 0 0 64 hardmining 2 0.000005 0.4 0 32.0"

    # "1 0 0 0 1 64 hardmining 2 0.000001 0.2 0 64.0"   
    # "1 0 0 0 1 64 hardmining 2 0.000001 0.3 0 64.0" 
    # "1 1 1 0 1 64 hardmining 2 0.000001 0.2 0 64.0"   
    # "1 0 0 0 0 64 hardmining 2 0.000005 0.2 0 64.0"
    # "1 0 0 0 0 64 hardmining 2 0.000005 0.4 0 64.0"
    # "1 1 1 1 1 64 hardmining 2 0.000001 0.2 0 64.0"   
    # "1 1 0 0 0 64 hardmining 2 0.000001 0.2 0 64.0"   
    # "1 1 1 0 1 64 hardmining 2 0.000005 0.2 0 64.0"   
    # "1 1 1 1 1 64 hardmining 2 0.000005 0.2 0 64.0"  


    # "1 0 0 0 1 64 hardmining 3 0.000001 0.2 0 64.0"    
    # "1 1 1 1 1 64 hardmining 3 0.000001 0.2 0 64.0" 
    # "1 0 0 0 1 64 hardmining 3 0.000001 0.3 0 64.0"    
    # "1 1 1 1 1 64 hardmining 3 0.000001 0.3 0 64.0" 







    # "1 0 0 0 1 64 hardmining 2 0.000005 0.2 0" 
    # "0 1 1 1 1 64 hardmining 2 0.0001 0.2"     

    # "1 1 1 1 1 64 hardmining 1 0.00001 0.1"   
    # "1 0 0 1 0 32 hardmining 1 0.000005 0.3"  
    # "0 0 0 0 1 16 hardmining 1 0.000001 0.4"  
    
    # "1 1 1 1 1 64 hardmining 0 0.00001 0.2"   
    # "0 0 1 1 0 32 hardmining 0 0.0001 0.1"    
    # "1 0 1 0 0 64 hardmining 0 0.000001 0.2"  
    
    # "1 1 1 1 1 16 hardmining 1 0.0001 0.5"    
    # "1 0 0 1 0 64 hardmining 0 0.00005 0.3"   
    # "0 1 1 0 1 32 hardmining 1 0.000005 0.4"


    # "1 1 0 0 1 128 hardmining 0 0.000005 0.5" 
    # "0 1 1 0 1 128 hardmining 1 0.00005 0.2"  
    # "1 1 1 1 1 128 hardmining 2 0.000001 0.1" 
    # "0 1 0 1 0 128 hardmining 2 0.00005 0.1"  
)


RESULTS_FILE="batch_run_results2.txt"
> "$RESULTS_FILE" 


for SET in "${PARAMETER_SETS[@]}"; do
    read P_CLIP P_SEG P_MIDAS P_DPT P_RES P_MOB P_EFF P_VIT P_GATE P_BATCH P_TYPE P_HEAD P_LR P_MARGIN P_ALPHA <<< "$SET"

    echo "--- Starting run with Models: Clip=$P_CLIP Seg=$P_SEG Midas=$P_MIDAS Dpt=$P_DPT Res=$P_RES Mob=$P_MOB Eff=$P_EFF Vit=$P_VIT ---"

    COMMAND_TO_RUN="python3 -m $PYTHON_SCRIPT \
    --clip $P_CLIP \
    --segformer $P_SEG \
    --midas $P_MIDAS \
    --dpt $P_DPT \
    --resnet $P_RES \
    --mobilenet $P_MOB \
    --efficientnet $P_EFF \
    --vit $P_VIT \
    --gate $P_GATE \
    --batch $P_BATCH \
    --train_type $P_TYPE \
    --big_fusion_head $P_HEAD \
    --lr $P_LR \
    --margin $P_MARGIN \
    --alpha $P_ALPHA"

    echo "$COMMAND_TO_RUN"
    
    OUTPUT=$($COMMAND_TO_RUN 2>&1 | tee /dev/tty | tail -n 1)
    EXIT_CODE=${PIPESTATUS[0]}

    FINAL_OUTPUT_LINE="$OUTPUT"

    echo "Run clip=$P_CLIP seg=$P_SEG midas=$P_MIDAS dpt=$P_DPT res=$P_RES mob=$P_MOB eff=$P_EFF vit=$P_VIT gate=$P_GATE batch=$P_BATCH type=$P_TYPE head=$P_HEAD lr=$P_LR margin=$P_MARGIN alpha=$P_ALPHA: Status Code $EXIT_CODE, Result: $FINAL_OUTPUT_LINE" >> "$RESULTS_FILE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Run finished **successfully**."
    else
        echo "Run **failed**. Stopping batch."
    fi
    echo "---"
done

echo "All runs in the batch completed."
echo "Final results collected in: **$RESULTS_FILE**"