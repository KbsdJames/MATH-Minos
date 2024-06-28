for LR in "Learning Rate List"; do
    OUTPATH="mathmatical_rm_pointwise_bsz512_warmup_$LR"
    bash run_prm.sh "$LR" "$OUTPATH"
    bash inference.sh \
        #CKPTS_PATH
        #GSM8k_OUTPATH \
        #MATH_OUTPATH \
done
