from setuptools import setup, find_packages
from distilkobert import __version__

with open('requirements.txt') as f:
    require_packages = [line[:-1] if line[-1] == '\n' else line for line in f]

setup(name='distilkobert',
      version=__version__,
      url='https://github.com/monologg/DistilKoBERT',
      license='Apache-2.0',
      author='Jangwon Park',
      author_email='adieujw@gmail.com',
      description='Distillation of KoBERT',
      packages=find_packages(exclude=['distillation', 'docker', 'nsmc']),
      long_description=open('../README.md', 'r', encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      python_requires='>=3',
      zip_safe=False,
      include_package_data=True,
      classifiers=(
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.6',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ),
      install_requires=require_packages,
      keywords="distillation kobert bert pytorch transformers lightweight"
      )
