model_name="baseline"
epoch=300

echo "python main.py --phase valid --model checkpoint/${model_name}/ep-${epoch}.pt &> logs/valid_${model_name}_ep${epoch}.log"
python main.py --phase valid --model checkpoint/${model_name}/ep-${epoch}.pt &> logs/valid_${model_name}_ep${epoch}.log
