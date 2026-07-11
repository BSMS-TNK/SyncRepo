import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from summarize_logs import main, summarize_log


class SummarizeLogsTest(unittest.TestCase):
    def test_summarize_log_extracts_metadata_best_and_final_metrics(self):
        content = """\
[00:00:00.000] python train.py --dataset fundus --lb_domain 1 --lb_num 40 --save_name fundus_dm1_lb40_full --gpu 0 --model MedSAM
[00:00:00.001] Namespace(dataset='fundus', save_name='fundus_dm1_lb40_full', max_iterations=30000, num_eval_iter=500, domain_num=4, lb_domain=1, lb_num=40, model='MedSAM')
[00:10:00.000] iteration 500 : loss : 0.377457, sam_sup_loss : 0.100733, sam_unsup_loss : 0.257611, unet_sup_loss : 0.269560, unet_unsup_loss : 0.582658, cons_w : 0.007820, mask_ratio : 0.893574, sd:0.672295,0.863125,ud:0.487320,0.651552,d:0.629366,0.868507,s_m_r:0.827062,0.613741,0.523415
[00:10:00.001] sam_ulb_cup_dice:0.672295, sam_ulb_disc_dice:0.863125, unet_ulb_cup_dice:0.487320, unet_ulb_disc_dice:0.651552, ulb_cup_dice:0.629366, ulb_disc_dice:0.868507
[00:10:00.002] test unet model
[00:10:01.000] epoch 1 : loss : 0.000000
\tval_cup_dice: 0.595408, val_disc_dice: 0.803349,
[00:10:01.001] val_cup_best_dice: 0.595408 at 500 iter, val_disc_best_dice: 0.803349 at 500 iter, val_best_avg_dice: 0.699379 at 500 iter, cup_dice: 0.595408, disc_dice: 0.803349
[00:10:01.002] test sam model
[00:10:01.500] domain1 epoch 1 : loss : 0.000000
\tval_cup_dice: 0.900000, val_disc_dice: 0.950000,
[00:10:02.000] epoch 1 : loss : 0.000000
\tval_cup_dice: 0.826944, val_disc_dice: 0.929697,
[00:10:02.001] stu_val_cup_best_dice: 0.826944 at 500 iter, stu_val_disc_best_dice: 0.929697 at 500 iter, val_best_avg_dice: 0.878321 at 500 iter, cup_dice: 0.826944, disc_dice: 0.929697
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.txt"
            path.write_text(content)

            row = summarize_log(path, Path(tmpdir))

        self.assertEqual(row["dataset"], "fundus")
        self.assertEqual(row["save_name"], "fundus_dm1_lb40_full")
        self.assertEqual(row["last_iteration"], 500)
        self.assertAlmostEqual(row["unet_best_avg_dice"], 0.699379)
        self.assertEqual(row["unet_best_avg_iter"], 500)
        self.assertAlmostEqual(row["sam_best_avg_dice"], 0.878321)
        self.assertEqual(row["sam_best_avg_iter"], 500)
        self.assertAlmostEqual(row["unet_final_avg_dice"], 0.6993785)
        self.assertAlmostEqual(row["sam_final_avg_dice"], 0.8783205)
        self.assertAlmostEqual(row["last_mask_ratio"], 0.893574)

    def test_main_writes_summary_to_output_file_without_printing_rows(self):
        content = """\
[00:00:00.000] python train.py --dataset BUSI --lb_domain 1 --lb_num 40 --save_name BUSI_dm1_lb40_full --model MedSAM
[00:00:01.000] iteration 500 : loss : 0.377457, mask_ratio : 0.893574
[00:00:02.000] test unet model
[00:00:03.000] epoch 1 : loss : 0.000000
\tval_base_dice: 0.496102,
[00:00:04.000] val_best_avg_dice: 0.496102 at 500 iter, base_dice: 0.496102
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            log_path = model_dir / "BUSI" / "train" / "BUSI_dm1_lb40_full" / "log.txt"
            output_path = Path(tmpdir) / "summary.json"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(content)

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main([
                    "--model-dir",
                    str(model_dir),
                    "--format",
                    "json",
                    "--output",
                    str(output_path),
                ])

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn('"dataset": "BUSI"', output_path.read_text())


if __name__ == "__main__":
    unittest.main()
