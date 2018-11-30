model_name="baseline"
epoch=300

echo "python main.py --phase test --model checkpoint/${model_name}/ep-${epoch}.pt &> logs/test_${model_name}_ep${epoch}.log"
python main.py --phase test --model checkpoint/${model_name}/ep-${epoch}.pt &> logs/test_${model_name}_ep${epoch}.log
