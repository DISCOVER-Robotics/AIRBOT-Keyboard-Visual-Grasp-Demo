"""Tests for endpointing without microphone, FunASR, or hardware."""

import unittest

from voice_asr_worker import SpeechEndpoint


class SpeechEndpointTest(unittest.TestCase):
    def test_silence_before_speech_does_not_stop(self):
        endpoint = SpeechEndpoint(0.01, 0.8)
        for _ in range(20):
            self.assertFalse(endpoint.feed(0.001, 0.1))
        self.assertFalse(endpoint.has_speech)

    def test_short_noise_is_not_valid_speech(self):
        endpoint = SpeechEndpoint(0.01, 0.8)
        endpoint.feed(0.02, 0.1)
        for _ in range(8):
            endpoint.feed(0.001, 0.1)
        self.assertFalse(endpoint.has_speech)

    def test_speech_stops_after_configured_silence(self):
        endpoint = SpeechEndpoint(0.01, 0.8)
        self.assertFalse(endpoint.feed(0.02, 0.1))
        self.assertFalse(endpoint.feed(0.02, 0.1))
        for _ in range(7):
            self.assertFalse(endpoint.feed(0.001, 0.1))
        self.assertTrue(endpoint.feed(0.001, 0.1))
        self.assertTrue(endpoint.has_speech)

    def test_new_speech_resets_silence_timer(self):
        endpoint = SpeechEndpoint(0.01, 0.8)
        endpoint.feed(0.02, 0.2)
        for _ in range(5):
            endpoint.feed(0.001, 0.1)
        endpoint.feed(0.02, 0.1)
        for _ in range(7):
            self.assertFalse(endpoint.feed(0.001, 0.1))
        self.assertTrue(endpoint.feed(0.001, 0.1))


if __name__ == "__main__":
    unittest.main()
