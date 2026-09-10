import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from remote_client import _poll_until_done, RemoteTranscribeError, remote_transcribe
from remote_job import JobLock, StatusWriter


class RemoteTests(unittest.TestCase):
    def test_resume_downloads_without_uploading(self):
        with tempfile.TemporaryDirectory() as temp:
            wav = Path(temp) / "meeting.wav"
            wav.touch()
            endpoint = Mock()
            endpoint.download_json.return_value = {
                "stage": "done", "request_id": "existing", "language": "en"
            }
            endpoint.download.return_value = b"transcript"
            with patch("remote_client.RemoteEndpoint", return_value=endpoint), \
                 patch("remote_client.prepare_audio_for_upload") as prepare:
                result = remote_transcribe(wav, "http://host", resume=True, poll_interval=0)
            self.assertEqual(wav.with_suffix('.txt').read_text(), "transcript")
            self.assertEqual(len(result['outputs']), 3)
            endpoint.upload_file.assert_not_called()
            endpoint.upload_bytes.assert_not_called()
            prepare.assert_not_called()

    def test_heartbeat_keeps_progress_and_terminal_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'status.json'
            status = StatusWriter(path, 'job', request_id='current')
            status.update('diarizing', progress=0.6, force=True)
            before = json.loads(path.read_text())
            with patch('remote_job.time.time', return_value=before['updated_at'] + 10):
                status.pulse()
            after = json.loads(path.read_text())
            self.assertGreater(after['updated_at'], before['updated_at'])
            self.assertEqual(after['progress'], 0.6)
            status.update('done', force=True)
            terminal = path.read_bytes()
            status.pulse()
            self.assertEqual(path.read_bytes(), terminal)

    def poll(self, endpoint):
        return _poll_until_done(endpoint, {"status": "status", "log": "log"},
                                Mock(), Mock(), 0, 100, "current")

    def test_old_success_is_not_returned_for_new_submission(self):
        endpoint = Mock()
        endpoint.download_json.side_effect = [
            {"stage": "done", "request_id": "old"},
            {"stage": "done", "request_id": "current", "language": "en"},
        ]
        self.assertEqual(self.poll(endpoint)["language"], "en")
        self.assertEqual(endpoint.download_json.call_count, 2)

    def test_current_failure_is_reported(self):
        endpoint = Mock()
        endpoint.download_json.return_value = {
            "stage": "error", "request_id": "current", "error": "Model unavailable",
        }
        with self.assertRaisesRegex(RemoteTranscribeError, "Model unavailable"):
            self.poll(endpoint)

    def test_queue_heartbeat_prevents_stall_timeout(self):
        endpoint = Mock()
        endpoint.download_json.side_effect = [
            {"stage": "queued", "request_id": "current", "updated_at": i}
            for i in range(4)
        ] + [{"stage": "done", "request_id": "current"}]
        ticks = iter(range(100))
        with patch("remote_client.time.time", side_effect=lambda: next(ticks)), \
             patch("remote_client.STALL_TIMEOUT", 3):
            self.assertEqual(self.poll(endpoint)["stage"], "done")

    def test_host_status_preserves_request_id(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "status.json"
            status = StatusWriter(path, "job", request_id="current")
            for stage in ("received", "queued", "done"):
                status.update(stage, force=True)
                self.assertEqual(json.loads(path.read_text())["request_id"], "current")

    def test_waiting_job_reports_each_poll(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lock"
            path.touch()
            on_wait = Mock()
            polls = []

            def advance(_):
                polls.append(True)
                if len(polls) == 3:
                    path.unlink()

            with patch("remote_job._lock_path", return_value=path), \
                 patch("remote_job.time.sleep", side_effect=advance):
                with JobLock(on_wait=on_wait):
                    self.assertEqual(on_wait.call_count, 3)


if __name__ == "__main__":
    unittest.main()
