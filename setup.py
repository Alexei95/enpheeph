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
        python_requires=">=3.8",
        zip_safe=False,
        # dependencies
        install_requires=[
            "setuptools >= 58.0",
        ],
        extras_require={
            "full": [
                "cupy >= 9.0.0",
                "norse >= 0.0.7",
                "numpy >= 1.19",
                "pytorch-lightning >= 1.5",
                "sqlalchemy >= 1.4.20",
                "torch >= 1.8, < 1.10",
            ],
        },
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
