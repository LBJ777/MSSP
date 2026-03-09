CUDA_VISIBLE_DEVICES=1 conda run -n aigc python experiments/step_b_1a_psd_diagnostic.py \
    --data_dir /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/data/step_a \
    --model_path /data/lizihao/AIGC/AIGCDetectBenchmark-main/weights/preprocessing/256x256_diffusion_uncond.pt \
    --num_samples 50 \
    --output_dir ./results/step_b_1a \
    --device cuda \
    --batch_size 4 \
    --n_freq_bins 64 \
    --top_k 20 2>&1