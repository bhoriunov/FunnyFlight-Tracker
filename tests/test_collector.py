import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

import adsb_data
import dataset_store
import pipeline


class CollectorTests(unittest.TestCase):
    def response(self, payload):
        return patch.object(adsb_data.requests, 'get', return_value=Mock(json=lambda: payload))

    def test_empty_response_scores_with_stable_schema(self):
        with self.response({'ac': []}):
            result = pipeline.score_live_dataset(pipeline.build_live_dataset())
        self.assertTrue(result.empty)
        self.assertTrue({'hex', 'rarity_score', 'anomaly_score', 'model_aircraft'} <= set(result.columns))

    def test_missing_optional_fields_and_ground_altitude(self):
        with self.response({'ac': [{'hex': 'abc123', 't': 'A320', 'flight': ' dlh123 ', 'alt_baro': 'ground'}]}):
            result = pipeline.score_live_dataset(pipeline.build_live_dataset())
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['hex'], 'ABC123')
        self.assertEqual(result.iloc[0]['operator_code'], 'DLH')
        self.assertEqual(result.iloc[0]['alt_baro'], 0)
        self.assertTrue(pd.isna(result.iloc[0]['registration']))
        self.assertTrue(pd.notna(result.iloc[0]['rarity_score']))

    def test_missing_required_fields_are_filtered(self):
        records = [{'hex': 'abc123', 't': 'A320'}, {'t': 'A320', 'flight': 'DLH123'}, {'hex': 'abc123', 'flight': 'DLH123'}, {'hex': 'abc123', 't': 'GND', 'flight': 'DLH123'}]
        with self.response({'ac': records}):
            self.assertTrue(pipeline.score_live_dataset(pipeline.build_live_dataset()).empty)

    def test_malformed_response_is_explicit_failure(self):
        for payload in ({}, None, {'ac': None}, {'ac': [None]}):
            with self.subTest(payload=payload), self.response(payload):
                with self.assertRaises(adsb_data.AdsbFetchError):
                    adsb_data.fetch_adsb_data()

    def test_http_and_json_failures(self):
        for response in (Mock(raise_for_status=Mock(side_effect=requests.HTTPError('503'))), Mock(json=Mock(side_effect=ValueError('invalid JSON')))):
            with patch.object(adsb_data.requests, 'get', return_value=response):
                with self.assertRaises(adsb_data.AdsbFetchError):
                    adsb_data.fetch_adsb_data()

    def test_timeout_preserves_files(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / 'snapshot.csv'
            training = Path(folder) / 'training.csv'
            snapshot.write_text('previous snapshot\n')
            training.write_text('previous training\n')
            with patch.object(pipeline, 'SNAPSHOT_PATH', snapshot), patch.object(dataset_store, 'TRAINING_PATH', training), patch.object(adsb_data.requests, 'get', side_effect=requests.Timeout('test')), contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(pipeline.run_once())
            self.assertEqual(snapshot.read_text(), 'previous snapshot\n')
            self.assertEqual(training.read_text(), 'previous training\n')

    def test_success_then_empty_preserves_history_clears_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / 'snapshot.csv'
            training = Path(folder) / 'training.csv'
            with patch.object(pipeline, 'SNAPSHOT_PATH', snapshot), patch.object(dataset_store, 'TRAINING_PATH', training), contextlib.redirect_stdout(io.StringIO()):
                with self.response({'ac': [{'hex': '123456', 't': 'A320', 'flight': 'DLH123'}]}):
                    self.assertTrue(pipeline.run_once())
                    self.assertTrue(pipeline.run_once())
                self.assertEqual(len(pd.read_csv(training)), 1)
                previous = training.read_bytes()
                with self.response({'ac': []}):
                    self.assertTrue(pipeline.run_once())
                self.assertTrue(pd.read_csv(snapshot).empty)
                self.assertEqual(training.read_bytes(), previous)

    def test_model_loads_outside_project_directory(self):
        root = str(Path(pipeline.__file__).resolve().parent)
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, '-B', '-c', 'import sys; sys.path.insert(0, sys.argv[1]); from predict import score_plane; print(score_plane("A320", "DLH")["rarity_score"])', root], cwd=folder, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_interval_retries_and_keyboard_interrupt_stops(self):
        with patch.object(sys, 'argv', ['pipeline.py', '--interval', '60']), patch.object(pipeline, 'run_once', side_effect=[False, True]) as run, patch.object(pipeline.time, 'sleep', side_effect=[None, KeyboardInterrupt]) as sleep, contextlib.redirect_stdout(io.StringIO()):
            pipeline.main()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(sleep.call_args.args, (60,))

    def test_one_shot_failure_exit_code(self):
        with patch.object(sys, 'argv', ['pipeline.py']), patch.object(pipeline, 'run_once', return_value=False):
            with self.assertRaises(SystemExit) as error:
                pipeline.main()
        self.assertEqual(error.exception.code, 1)

    def test_invalid_interval(self):
        with patch.object(sys, 'argv', ['pipeline.py', '--interval', '0']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                pipeline.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
