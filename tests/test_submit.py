from unittest import TestCase

import requests
import requests_mock
from mock import patch

from solver.submit import submit
from solver.types import Point, Solution


class TestSubmit(TestCase):
    @requests_mock.Mocker()
    @patch("solver.submit.internal_run")
    @patch("solver.submit.get_api_token")
    def test_submit(self, requests_mock, token_mock, run_mock):
        problem_id = 1
        expected_identifier = "123"

        run_mock.return_value = Solution([Point(1, 2), Point(2, 1)])
        token_mock.return_value = "fake_token"
        requests_mock.register_uri(
            "POST",
            f"https://poses.live/api/problems/{problem_id}/solutions",
            [{"text": f'{{ "id": "{expected_identifier}" }}'}],
        )

        response = submit(problem_id)
        self.assertEqual(response.id, expected_identifier)

    @requests_mock.Mocker()
    @patch("solver.submit.internal_run")
    @patch("solver.submit.get_api_token")
    def test_submit_fails(self, requests_mock, token_mock, run_mock):
        problem_id = 2
        requests_mock.register_uri(
            "POST",
            f"https://poses.live/api/problems/{problem_id}/solutions",
            [{"status_code": 500}],
        )
        run_mock.return_value = Solution([Point(1, 2), Point(2, 1)])
        token_mock.return_value = "fake_token"

        with self.assertRaises(requests.exceptions.HTTPError):
            submit(problem_id)
