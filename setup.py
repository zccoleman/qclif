from setuptools import setup
from pathlib import Path

here = Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")



setup(
    name='qclif',
    version=0.1,
    description='Clifford gates for qudits',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zccoleman/qclif"
)