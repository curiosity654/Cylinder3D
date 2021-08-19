name=cylinder_asym_networks
time=$(date "+%Y%m%d-%H%M%S")
python -u train_cylinder_asym.py -y config/semantickitti_baseline.yaml
2>&1 | tee logs_dir/${time}_${name}_logs_tee.txt