from setuptools import setup, find_packages

setup(
    name='kickstarter_breakdown',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'statsmodels',
        'seaborn',
        'matplotlib',
        'scipy',
        'plotly'
    ],
    author='Jaden Earl',
    description='A package for kickstarter data cleaning and analysis',
    url='https://github.com/yourusername/your_package'
)
