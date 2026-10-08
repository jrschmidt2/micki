"""The examples in docs/ run and print what the documentation shows.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import contextlib
import io
import os
import re
import unittest
import warnings

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'docs')


def code_and_output(filename):
    """The first ```python block of a page that is directly followed by a
    plain ``` block (its printed output), and that output."""
    with open(os.path.join(DOCS, filename)) as f:
        text = f.read()
    block = r'((?:(?!```).)*)```'
    match = re.search(r'```python\n' + block + r'\s*```\n' + block, text,
                      re.S)
    return match.group(1), match.group(2)


class DocsTest(unittest.TestCase):

    def test_user_guide_and_analysis_examples(self):
        # the analysis example continues the user guide's example
        namespace = {}
        for page in ('user-guide.md', 'analysis.md'):
            code, expected = code_and_output(page)
            out = io.StringIO()
            with warnings.catch_warnings(), contextlib.redirect_stdout(out):
                warnings.simplefilter('ignore')
                exec(code, namespace)
            self.assertEqual(out.getvalue(), expected, page)


if __name__ == '__main__':
    unittest.main()
