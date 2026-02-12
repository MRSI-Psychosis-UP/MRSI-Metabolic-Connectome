from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

def read_requirements():
    with open(ROOT / "requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mrsitoolbox",
    version="1.0.0",
    description="Analysis toolbox for MRSI data",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Federico Lucchetti",
    author_email="federico.lucchetti@unil.ch",
    url="https://github.com/MRSI-Psychosis-UP/Metabolic-Connectome.git",
    python_requires=">=3.8",
    packages=find_packages(include=["mrsitoolbox", "mrsitoolbox.*"]),
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        "mrsitoolbox.graphplot": ["cmaps/*.cmap", "cmaps/README"],
        "mrsitoolbox.connectomics": ["data/*.json"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: Other/Proprietary License",
    ],
)
