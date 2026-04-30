#!/bin/bash
# Start both API and bot in the same Railway service
python run_api.py &
python run_bot.py
