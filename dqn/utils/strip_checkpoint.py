"""Shrink a training checkpoint to the Q-network weights only (fp16), enough to record a demo.
usage: python strip_checkpoint.py checkpoints/best.pth best_small.pth
(full checkpoints also hold the target net and Adam state, ~5x larger)"""
import sys, torch

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu")
sd = {k: v.half() for k, v in ckpt["policy_net"].items()}
torch.save({"policy_net": sd, "epsilon": ckpt.get("epsilon"), "step_cnt": ckpt.get("step_cnt")}, dst)
print(f"{src} -> {dst}")
