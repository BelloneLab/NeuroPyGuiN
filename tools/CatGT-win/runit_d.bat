@echo off
@setlocal enableextensions
@cd /d "%~dp0"

:: Custom local parameters for digital input extraction from trial1
set LOCALARGS=-dir=B:\NPX\rawData\VTA_NPX\29540\1 -run=trial1 -g=0 -t=0 ^
-ni -digout=7,1

:: If no parameters passed, use LOCALARGS; otherwise, pass the provided arguments
if [%1]==[] (set ARGS=%LOCALARGS%) else (set ARGS=%*)

:: Run CatGT
%~dp0CatGT %ARGS%
