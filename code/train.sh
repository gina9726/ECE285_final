model_name="baseline"

echo "main.py --phase train --save_model ${model_name} &> logs/${model_name}.log"
python main.py --phase train --save_model ${model_name} &> logs/${model_name}.log
