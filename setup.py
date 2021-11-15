# -*- coding: utf-8 -*-
# import importlib
import setuptools

# about = importlib.util.

if __name__ == "__main__":
    setuptools.setup(
        name="enpheeph",
        description="",
        version="0.0.1",
        # long_description=,
        # long_description_content_type=,
        # keywords=,
        # author=,
        # author_email=,
        # url="https://github.com/psf/black",
        # project_urls={
        #     "Changelog": "https://github.com/psf/black/blob/main/CHANGES.md",
        # },
        # license="MIT",
        packages=setuptools.find_packages(where="src"),
        # if packages are inside src recursively
        package_dir={"": "src"},
        # for py.typed and PEP-517 compatibility
        # package_data={
        #     "blib2to3": ["*.txt"],
        #     "black": ["py.typed"],
        #     "black_primer": ["primer.json"],
        # },
        python_requires=">=3.9",
        zip_safe=False,
        # dependencies
        # install_requires=[
        #     "click>=7.1.2",
        #     "platformdirs>=2",
        #     "tomli>=0.2.6,<2.0.0",
        #     "typed-ast>=1.4.2; python_version < '3.8'",
        #     "regex>=2020.1.8",
        #     "pathspec>=0.9.0, <1",
        #     "dataclasses>=0.6; python_version < '3.7'",
        #     "typing_extensions>=3.10.0.0",
        #     # 3.10.0.1 is broken on at least Python 3.10,
        #     # https://github.com/python/typing/issues/865
        #     "typing_extensions!=3.10.0.1; python_version >= '3.10'",
        #     "mypy_extensions>=0.4.3",
        # ],
        # extras_require={
        #     "d": ["aiohttp>=3.7.4"],
        #     "colorama": ["colorama>=0.4.3"],
        #     "python2": ["typed-ast>=1.4.3"],
        #     "uvloop": ["uvloop>=0.15.2"],
        #     "jupyter": ["ipython>=7.8.0", "tokenize-rt>=3.2.0"],
        # },
        # test_suite="tests.test_black",
        # classifiers=[
        #     "Development Status :: 4 - Beta",
        #     "Environment :: Console",
        #     "Intended Audience :: Developers",
        #     "License :: OSI Approved :: MIT License",
        #     "Operating System :: OS Independent",
        #     "Programming Language :: Python",
        #     "Programming Language :: Python :: 3.6",
        #     "Programming Language :: Python :: 3.7",
        #     "Programming Language :: Python :: 3.8",
        #     "Programming Language :: Python :: 3.9",
        #     "Programming Language :: Python :: 3 :: Only",
        #     "Topic :: Software Development :: Libraries :: Python Modules",
        #     "Topic :: Software Development :: Quality Assurance",
        # ],
        # entry_points={
        #     "console_scripts": [
        #         "black=black:patched_main",
        #         "blackd=blackd:patched_main [d]",
        #         "black-primer=black_primer.cli:main",
        #     ]
        # },
    )
