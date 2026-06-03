@echo off
cd /d "%~dp0"
python dubois_anaglyph_app.py
if errorlevel 1 pause
