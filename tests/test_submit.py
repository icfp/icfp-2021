from unittest import TestCase
from solver.submit import submit_problem


class TestSubmit(TestCase):
    def test_submit(self):
        submit_problem(1)
