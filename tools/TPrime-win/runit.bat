
:: You can call TPrime three ways:
::
:: 1) > TPrime cmd-line-parameters
:: 2) > runit.bat cmd-line-parameters
:: 3a) Edit parameters in runit.bat, then call it ...
:: 3b) > runit.bat
::
:: This script effectively says:
:: "If there are no parameters sent to runit.bat, call TPrime
:: with the parameters hard coded here, else, pass all of the
:: parameters through to TPrime."
::

@echo off
@setlocal enableextensions
@cd /d "%~dp0"

set LOCALARGS=-syncperiod=1.0 ^
-tostream=Y:\tptest\time_trans_01_g0_tcat.imec0.ap.SY_301_6_500.txt ^
-fromstream=7,Y:\tptest\time_trans_01_g0_tcat.nidq.XA_0_500.txt ^
-events=7,Y:\tptest\time_trans_01_g0_tcat.nidq.XA_1_7700.txt,Y:\tptest\out.txt

if [%1]==[] (set ARGS=%LOCALARGS%) else (set ARGS=%*)

%~dp0TPrime %ARGS%

