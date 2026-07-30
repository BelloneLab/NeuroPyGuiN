
Purpose:
+ Map from-stream event times into to-stream times.
Run messages are appended to TPrime.log in the current working directory.

Install:
1) Copy TPrime-linux to your machine, cd into folder.
2) If needed, > chmod +x install.sh
3) > ./install.sh
4) Read notes in runit.sh wrapper script (required).

Compatibility:
- Included libraries are from Ubuntu 16.04 (Xenial).
- Tested with Ubuntu 20.04 and 20.10.
- Tested with Scientific Linux 7.3.
- Tested with Oracle Linux Server 8.3.
- Let me know if it runs on other distributions.

Usage:
>runit.sh -syncperiod=1.0 -tostream=path/edgefile.txt <from_data> [ options ]

# Enclosing whole parameter list in quotes is recommended, like this:
# >runit.sh '-syncperiod=1.0 -tostream=path/edgefile.txt <from_data> [ options ]'

From_data:
-fromstream=5,path/edgefile.txt                        ;stream_index,edge_times
-events=5,path/in_eventfile.txt,path/out_eventfile.txt ;stream_index,in_event_times,out_event_times

Options:
-offsets=path/offsetFile.txt

Notes:
- The sync pulser period can be read from SpikeGLX metadata files 'syncSourcePeriod' (enter value as recorded in file).
- Streams are characterized by a file of pulser edge times.
- Obtain edge files by running CatGT to extract the sync channel from each stream of interest.
- Each from-stream needs a zero-based stream_index that you assign arbitrarily (5 in this example); it merely links event files with their source streams.
- You can list as many -events entries as needed, so you can do all translation work in one call.

- Events files can be txt or Kilosort/KS2 npy files.
    + Input and output file types need not match.
    + Times are seconds, ** NOT SAMPLES **.
    + Times are relative to the start of a file.
    + Times must be sorted in ascending order.

- For npy input Events files:
    + We expect an array of doubles with dim {n-values}.

- For txt input Events files:
    + One event time per line.
    + Time must be only item (or first item) on line.
    + Times should be printed to 6 decimal digits precision (30000.000000).

- Optional parameter -offsets=path/offsetFile.txt names a CatGT ct_offsets or sc_offsets file which lists the starting sample of each composing file in the concatenated whole. These data can help improve mapping accuracy for events occuring very near the boundary of two composing files. If a first pass of CatGT was run with -zerofillmax or a supercat run did not use edge trimming, then sync edge spacing can become irregular in the boundary zones. The offset data can resolve ambiguities.


Change Log
----------
Version 1.8
- Add -offsets option.

Version 1.7
- Fix an end-of-file parsing bug.

Version 1.6
- Fix potential crash.

Version 1.5
- Improved calling scripts.

Version 1.4
- Working/calling dir can be different from installed dir.
- Log file written to working dir.

Version 1.3
- Handle fortran-order npy files.
- Output npy headers are x64 size.

Version 1.2
- Events files can be txt or npy.

Version 1.1
- Add error messages for file existence.

Version 1.0
- Initial release.


