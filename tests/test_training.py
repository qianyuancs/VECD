from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from vecd import VECD
from vecd.data import Records, Episodes
from vecd.evaluation import evaluate, merge_shards, calibrate
from vecd.io import read_json, save_weights, write_json
from vecd.training import train


class TrainingSmoke(unittest.TestCase):
    def test_static_paired_resume_and_independent_eval(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(1)
            rows=[]
            for i,split in enumerate(("train","train","val","val","cal","cal")):
                mask=np.zeros((16,16),np.uint8);mask[4:12,4:12]=255
                Image.fromarray(mask).save(root/f"mask{i}.png")
                clip=rng.standard_normal((2,4,5,512)).astype(np.float16)
                dino=rng.standard_normal((2,32,32,384)).astype(np.float16)
                np.savez(root/f"feature{i}.npz",clip=clip,dino=dino)
                np.savez(root/f"labels{i}.npz",labels=np.zeros((2,4,16,16),np.uint8))
                rows.append(dict(id=str(i),group=str(i),split=split,mask=f"mask{i}.png",positive=["color"],
                                 negative=["size","shape","material"],feature=f"feature{i}.npz",static_labels=f"labels{i}.npz"))
            write_json(root/"records.json",dict(records=rows))
            model=VECD();model.base.state.queries.copy_(torch.randn(5,512))
            save_weights(model,root/"init.safetensors")
            from safetensors.torch import save_file
            save_file({f"prototype_{i}":torch.randn(2,512) for i in range(4)},str(root/"prototypes.safetensors"))
            write_json(root/"frames.json",[dict(record="0",time=0)])
            write_json(root/"train.json",[dict(support="0",query="1",family="color",mode="set",epoch=0),
                                           dict(support="0",query="1",family="color",mode="mask",epoch=0)])
            base=dict(seed=5,epochs=1,data_root=str(root),cache=str(root/"cache"),manifest=str(root/"records.json"),
                      initialization=str(root/"init.safetensors"),effective_batch=1,microbatch=1,workers=0)
            cfg=dict(base,stage="static",output=str(root/"static"),frames=str(root/"frames.json"),prototypes=str(root/"prototypes.safetensors"))
            write_json(root/"static.json",cfg)
            result=train(root/"static.json","cpu")
            self.assertEqual(result["step"],1)
            cfg=dict(base,stage="parent",output=str(root/"parent"),episodes=str(root/"train.json"))
            write_json(root/"parent.json",cfg)
            first=train(root/"parent.json","cpu",max_steps=1)
            self.assertTrue(first["stopped_after_smoke_budget"])
            second=train(root/"parent.json","cpu")
            self.assertEqual(second["step"],2)
            from vecd.io import load_weights
            model=load_weights(VECD(),root/"parent/final.safetensors")
            records=Records(root/"records.json",root,root/"cache")
            es=[dict(support="2",query="3",family="color",mode="set")]
            evaluate(model,records,es,{"set":.5},torch.device("cpu"),root/"eval")
            merged=merge_shards(records,es,root/"eval")
            self.assertTrue(merged["all_saved_masks_recounted"])
            with self.assertRaises(ValueError):
                calibrate(model,records,es,torch.device("cpu"))
            cal=calibrate(model,records,[dict(support="4",query="5",family="color",mode="set")],torch.device("cpu"))
            self.assertTrue(cal["source_calibration_only"])


if __name__ == "__main__":
    unittest.main()
