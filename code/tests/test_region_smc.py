import unittest

import torch

from utils.region_smc import blend_region_wise_probabilities


class RegionWiseSMCTest(unittest.TestCase):
    def test_returns_spatial_maps_without_gradients(self):
        sam_prob = torch.tensor(
            [[[[0.8, 0.3]], [[0.2, 0.7]]]], requires_grad=True
        )
        unet_teacher_prob = torch.tensor(
            [[[[0.7, 0.4]], [[0.3, 0.6]]]], requires_grad=True
        )
        unet_student_prob = torch.tensor(
            [[[[0.6, 0.5]], [[0.4, 0.5]]]], requires_grad=True
        )

        blended, self_conf, mutual_conf, alpha_map = (
            blend_region_wise_probabilities(
                sam_prob,
                unet_teacher_prob,
                unet_student_prob,
            )
        )

        self.assertEqual(blended.shape, (1, 2, 1, 2))
        self.assertEqual(self_conf.shape, (1, 1, 1, 2))
        self.assertEqual(mutual_conf.shape, (1, 1, 1, 2))
        self.assertEqual(alpha_map.shape, (1, 1, 1, 2))
        for tensor in (blended, self_conf, mutual_conf, alpha_map):
            self.assertFalse(tensor.requires_grad)
        for confidence_map in (self_conf, mutual_conf, alpha_map):
            self.assertTrue(torch.all(confidence_map >= 0.0))
            self.assertTrue(torch.all(confidence_map <= 1.0))
        torch.testing.assert_close(
            blended.sum(dim=1), torch.ones_like(blended[:, 0])
        )

    def test_agreement_moves_alpha_to_half(self):
        shared_prob = torch.tensor([[[[0.9]], [[0.1]]]])

        _, _, mutual_conf, alpha_map = blend_region_wise_probabilities(
            shared_prob,
            shared_prob,
            shared_prob,
        )

        torch.testing.assert_close(mutual_conf, torch.ones_like(mutual_conf))
        torch.testing.assert_close(alpha_map, torch.full_like(alpha_map, 0.5))

    def test_stable_confident_unet_is_preferred_during_disagreement(self):
        sam_prob = torch.tensor([[[[0.55]], [[0.45]]]])
        unet_prob = torch.tensor([[[[0.01]], [[0.99]]]])

        blended, _, _, alpha_map = blend_region_wise_probabilities(
            sam_prob,
            unet_prob,
            unet_prob,
        )

        self.assertGreater(alpha_map.item(), 0.5)
        self.assertEqual(blended.argmax(dim=1).item(), 1)

    def test_unstable_unet_is_deprioritized(self):
        sam_prob = torch.tensor([[[[0.99]], [[0.01]]]])
        unet_teacher_prob = torch.tensor([[[[0.45]], [[0.55]]]])
        unet_student_prob = sam_prob.clone()

        blended, self_conf, _, alpha_map = blend_region_wise_probabilities(
            sam_prob,
            unet_teacher_prob,
            unet_student_prob,
        )

        self.assertLess(self_conf.item(), 0.5)
        self.assertLess(alpha_map.item(), 0.5)
        self.assertEqual(blended.argmax(dim=1).item(), 0)

    def test_supports_multiclass_probability_maps(self):
        sam_prob = torch.softmax(torch.randn(2, 3, 4, 5), dim=1)
        unet_teacher_prob = torch.softmax(torch.randn(2, 3, 4, 5), dim=1)
        unet_student_prob = torch.softmax(torch.randn(2, 3, 4, 5), dim=1)

        blended, self_conf, mutual_conf, alpha_map = (
            blend_region_wise_probabilities(
                sam_prob,
                unet_teacher_prob,
                unet_student_prob,
            )
        )

        self.assertEqual(blended.shape, (2, 3, 4, 5))
        self.assertEqual(self_conf.shape, (2, 1, 4, 5))
        self.assertEqual(mutual_conf.shape, (2, 1, 4, 5))
        self.assertEqual(alpha_map.shape, (2, 1, 4, 5))


if __name__ == '__main__':
    unittest.main()
