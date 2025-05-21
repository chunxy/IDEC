source $MYDISK/miniconda3/etc/profile.d/conda.sh
conda activate system

python idec.py \
--dataset audio \
--n_clusters 1000 \
--batch_size 256 \
--n_z 10 \
--gamma 0.1 \
--update_interval 1 \
--tol 0.001 \
--lr 0.001 \