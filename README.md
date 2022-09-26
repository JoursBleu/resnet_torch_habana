This is a small sample about how to run a pytorch model on HABANA Gaudi.

The base code main.py is from https://github.com/pytorch/examples/tree/main/imagenet

The hpu version is main_hpu.py.



To run resnet50 on single card fp32:

python3 main_hpu.py -a resnet50 --use-hpu /software/datasets/imagenet_vm/

To run resnet50 on single card bf16:

python3 main_hpu.py -a resnet50 --use-hmp --bf16-config-path ./ops_bf16_Resnet.txt --fp32-config-path ./ops_fp32_Resnet.txt --use-hpu /software/datasets/imagenet_vm/

To run resnet50 on 8 cards fp32:

mpirun --allow-run-as-root -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings python3 main_hpu.py --world-size 8 -a resnet50 --use-hpu /software/datasets/imagenet_vm/

To run resnet50 on 8 cards bf16:

mpirun --allow-run-as-root -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings python3 main_hpu.py --world-size 8 -a resnet50 --use-hmp --bf16-config-path ./ops_bf16_Resnet.txt --fp32-config-path ./ops_fp32_Resnet.txt --use-hpu /software/datasets/imagenet_vm/


