#!/bin/bash
# Start API and bot in the same Railway service

# Railway assigns PORT — API must use it
# Bot connects to API on localhost with same port
export API_PORT=${PORT:-8000}
export API_BASE_URL="http://localhost:${API_PORT}"

echo "Starting API on port ${API_PORT}..."
python run_api.py &
API_PID=$!

# Wait for API to be ready before starting bot
echo "Waiting for API to start..."
sleep 5

# Check API is actually up
for i in 1 2 3 4 5; do
    if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        echo "API is ready."
        break
    fi
    echo "Waiting... attempt $i"
    sleep 3
done

echo "Starting Telegram bot..."
python run_bot.py

# If bot exits, kill API too
kill $API_PID
