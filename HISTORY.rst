=======
History
=======
2.4.2 (2022-28-12)
Fix HyperModel to support newer versions of numpy.

2.4.1 (2022-02-10)
Add support for new noise definitions in WidebandTimingModel.

2.4.0 (2022-02-10)
Use Timing Package (Tempo,Tempo2,Pint) definition of EQUAD. Enterprise has
broken backwards compatibility, and here we use the `tnequad` flag to switch on
the old definition.

2.3.4 (2021-11-02)
Fix phase shift seed caching issue.

2.3.3 (2021-10-04)
Fix bug in release build by adding ACE text file to MANIFEST.in.

2.3.2 (2021-10-04)
Fix bug in HyperModel when using save_runtime_info.

2.3.1 (2021-09-30)
Fix bugs associated with recent function additions. Added linting and mild PEP8
rules. Also removed older Python functionality which is no longer supported.

2.3.0 (2021-09-15)
Functionality added for NANOGrav 15yr dataset analyses.
Outlier analysis software moved into separate package.

2.2.0 (2021-08-10)
Version with outlier analysis.

0.9.1 (2021-05-06)
0.9.0 (2019-09-20)
------------------

* First release on PyPI.
