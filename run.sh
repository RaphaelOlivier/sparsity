
python train.py --model-type resnet18
python train.py --adv-train --ord-train inf --eps-train 0.0356 --lr-train 0.01 --iters-train 20 --model-type resnet18


ARGSMETRIC="--only-sparsity --num-samples 500 --num-cones 64 --ord-eval inf --eps-eval 0.0356 --lr-eval 0.01 --iters-eval 20 --batch-size 16"
ARGSOUTPUT=" 2>&1 | tee -a log/inf_log.txt"

eval "python eval.py $ARGSMETRIC --save-sparsity $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --adv-train --ord-train inf --eps-train 0.0356 --lr-train 0.01 --iters-train 20 $ARGSOUTPUT"

eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Rebuffi2021Fixing_70_16_cutmix_extra $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Gowal2020Uncovering_70_16_extra $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Rebuffi2021Fixing_70_16_cutmix_ddpm $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Rebuffi2021Fixing_28_10_cutmix_ddpm $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Kang2021Stable $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Sehwag2021Proxy $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Sehwag2021Proxy_R18 $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Huang2021Exploring $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Huang2021Exploring_ema $ARGSOUTPUT"
eval "python eval.py $ARGSMETRIC --save-sparsity --model-type Rade2021Helper_R18_ddpm $ARGSOUTPUT"
