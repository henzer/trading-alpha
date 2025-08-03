from setuptools import setup, find_packages

setup(
    name="trading-alpha",
    version="1.0.0",
    author="Claude & Henzer",
    description="Institutional Trading Framework for 20-30% Annual Returns",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "yfinance>=0.2.18",
        "vectorbt>=0.25.2", 
        "riskfolio-lib>=6.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "optuna>=3.4.0",
        "scikit-learn>=1.3.0",
        "numba>=0.57.0",
        "dash>=2.14.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)