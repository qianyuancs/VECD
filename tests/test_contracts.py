import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from vecd.data import Records, Episodes, validate_groups
from vecd.io import load_weights, save_weights, write_json
from vecd.losses import paired_loss, segmentation_loss
from vecd.metrics import confusion, measures
from vecd.model import VECD, StateReader, family_evidence, mask_selection


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reader_dimensions_and_unit_states(self):
        reader = StateReader()
        z = reader(torch.randn(2, 5, 512), torch.randn(2, 384), torch.randn(5, 512))
        self.assertEqual(tuple(z.shape), (2, 5, 512))
        self.assertTrue(torch.allclose(z.norm(dim=-1), torch.ones(2, 5), atol=1e-6))

    def test_family_energy_is_time_symmetric(self):
        q = F.normalize(torch.randn(1, 2, 8, 5, 512), dim=-1)
        s = F.normalize(torch.randn(1, 2, 8, 5, 512), dim=-1)
        a = family_evidence(q, s, torch.ones(1, 8))
        self.assertTrue(torch.equal(a, family_evidence(q.flip(1), s, torch.ones(1, 8))))
        self.assertTrue(torch.equal(a, family_evidence(q, s.flip(1), torch.ones(1, 8))))

    def test_empty_support_evidence_is_zero(self):
        q = torch.randn(1, 2, 8, 5, 512)
        s = torch.randn_like(q)
        out = family_evidence(q, s, torch.zeros(1, 8))
        self.assertEqual(float(out[:, :, 0].abs().sum()), 0.)

    def test_mask_area_is_read_directly(self):
        mask = torch.zeros(1, 16, 16);mask[:, :8] = 1
        selection = mask_selection(mask, (8, 8), 1).sigmoid()
        self.assertEqual(float(selection[:, :4].min()), 1.)
        self.assertEqual(float(selection[:, 4:].max()), 0.)
        with self.assertRaises(ValueError):
            mask_selection(mask-1, (8, 8), 1)

    def test_forward_cannot_accept_query_gt_or_attribute_id(self):
        keys = set(inspect.signature(VECD.forward).parameters)
        self.assertFalse(keys & {"target", "family", "query_mask", "caption", "operation_id"})

    def test_frozen_state_no_rgb_and_checkpoint_roundtrip(self):
        model = VECD()
        self.assertFalse(any(p.requires_grad for p in model.base.state.parameters()))
        self.assertFalse(any(".rgb." in k or ".detail." in k for k in model.state_dict()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.safetensors"
            save_weights(model, path)
            other = load_weights(VECD(), path)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, other.state_dict()[name]))

    def test_confusion_and_miou_definition(self):
        c = confusion(np.array([[1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]))
        self.assertEqual(c[:4].tolist(), [1, 1, 1, 1])
        self.assertAlmostEqual(measures(c)["binary_mIoU"], 1/3)
        self.assertAlmostEqual(measures(c)["precision"], .5)

    def test_loss_finite_and_mask_skips_selector_loss(self):
        out = {k:torch.randn(*shape, requires_grad=True) for k,shape in
               [("logits",(1,8,8)),("selection",(1,8,8)),("change",(1,2,8,8))]}
        batch = dict(target=torch.zeros(1,8,8), selection=torch.ones(1,8,8),
                     change=torch.zeros(1,2,8,8), valid=torch.ones(1,2,8,8),
                     mask_present=torch.ones(1,dtype=torch.bool))
        loss, terms = paired_loss(out, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(terms[1]),0.)
        self.assertEqual(float(out["selection"].grad.abs().sum()),0.)

    def test_scene_split_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_groups({"records":[dict(group="scene1",split="train"), dict(group="scene1",split="test")]})

    def test_nearest_resize_conventions_are_not_conflated(self):
        from vecd.io import resize_mask
        mask=np.zeros((3,5),np.uint8);mask[1,2]=1
        a=resize_mask(mask,8,"pil_nearest")
        b=resize_mask(mask,8,"torch_nearest")
        self.assertFalse(np.array_equal(a,b))

    def test_preparation_paths_and_cross_family_overlap(self):
        from PIL import Image
        from vecd.preparation import prepare_annotated_pairs, prepare_natural
        from vecd.io import resolve
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/"raw";raw.mkdir()
            for name in ("before", "after"):
                Image.fromarray(np.zeros((4,4,3),np.uint8)).save(raw/(name+".png"))
            masks=np.zeros((2,4,4),bool);masks[0,:2,:2]=True;masks[1,2:,2:]=True
            np.savez(raw/"before.npz",masks=masks,object_ids=np.arange(2))
            np.savez(raw/"after.npz",masks=masks[::-1],object_ids=np.arange(2))
            objects=[dict(id=0,color="red",size="small",shape="cube",material="metal"),
                     dict(id=1,color="blue",size="large",shape="sphere",material="rubber")]
            after=[dict(objects[0],color="blue"),dict(objects[1],shape="cube")]
            metadata=[dict(id="r1",group="scene",split="train",before="before.png",after="after.png",
                           objects_before=objects,objects_after=after,masks_before="before.npz",masks_after="after.npz",
                           edits=[dict(object_id=0,family="color"),dict(object_id=1,family="shape")])]
            write_json(root/"annotated.json",metadata)
            manifest=prepare_annotated_pairs(root/"annotated.json",raw,root/"prepared")
            self.assertTrue(resolve(root/"prepared",manifest["records"][0]["before"]).is_file())
            records=Records(root/"prepared/records.json",root/"prepared",root/"cache")
            target,change,valid=records.truth("r1","color")
            self.assertFalse(target.any());self.assertFalse(change.any());self.assertEqual(int(valid.sum()),8)
            Image.fromarray(np.ones((4,4),np.uint8)*255).save(raw/"edit.png")
            write_json(root/"natural.json",[dict(id="n1",group="image",split="train",before="before.png",
                        after="after.png",edit_mask="edit.png",positive=["color"],negative=["shape"])])
            natural=prepare_natural(root/"natural.json",raw,root/"natural")
            self.assertTrue(resolve(root/"natural",natural["records"][0]["after"]).is_file())


if __name__ == "__main__":
    unittest.main()
