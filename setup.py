import setuptools
import subprocess
import os

package_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)
# print(package_version)

if "-" in package_version:
    # when not on tag, git describe outputs: "1.3.3-22-gdf81228"
    # pip has gotten strict with version numbers
    # so change it to: "1.3.3+22.git.gdf81228"
    # See: https://peps.python.org/pep-0440/#local-version-segments
    v,i,s = package_version.split("-")
    # package_version = v + "+" + i + ".git." + s
    package_version = v 

# v_strs = package_version.split(".")
# print(v_strs)
# package_version = ".".join(v_strs[:-1]) + '.' + str(int(v_strs[-1])+1)
assert "-" not in package_version
assert "." in package_version

assert os.path.isfile("qsimkit/version.py")
with open("qsimkit/VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % package_version)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qsimkit",
    version=package_version,
    author="Jue XU",
    author_email="xujue@connect.hku.hk",
    description="Quantum Simulation Toolkit - Error bounds and Trotterization tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jue-Xu/Qsimkit",
    packages=setuptools.find_packages(),
    package_data={"qsimkit": ["VERSION"]},
    include_package_data=True,
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
    entry_points={"console_scripts": ["qsimkit = qsimkit.main:main"]},
    install_requires=[
        "qiskit >= 1.0.2",
        "qiskit-aer>=0.17.2",
        "jaxlib >= 0.6.2",
        "jax >= 0.6.2",
        "openfermion >= 1.5.1",
        "openfermionpyscf >= 0.5",
        "matplotlib >= 3.8.2",
        "numpy >= 1.23.5",
        "pandas >= 2.2.2",
        "scipy >= 1.12.0",
        "colorspace>=1.0.0",
        "multiprocess>=0.70.16",
    ],
)