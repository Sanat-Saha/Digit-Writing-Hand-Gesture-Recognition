@echo off
netsh wlan stop hostednetwork
netsh wlan set hostednetwork ssid=softcomputing
netsh wlan set hostednetwork key=user1234
netsh wlan start hostednetwork
:loop
python accelerometer.py
set /p val= continue[y/n]?:
if %val% == y goto loop