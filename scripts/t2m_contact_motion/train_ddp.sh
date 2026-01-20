EXP_NAME=$1
PORT=$2

if [ -z "$PORT" ]
then
    PORT=29500
fi

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} train_ddp.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=1000 \
            task=text_to_motion_contact_motion_gen \
            task.dataset.sigma=0.8 \
            task.train.batch_size=64 \
            task.train.max_steps=600000 \
            task.train.save_every_step=25000 \
            task.dataset.train_transforms=['RandomEraseLang','RandomEraseContact','NumpyToTensor'] \
           # +task.dataset.pred_contact_dir=map_pointmamba \
            #task.dataset.mix_train_ratio=0.5 \
            model=cmdm \
            model.arch='dit' \
            model.data_repr='h3d' \
            model.text_model.max_length=20
