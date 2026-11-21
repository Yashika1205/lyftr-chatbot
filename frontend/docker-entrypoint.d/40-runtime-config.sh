#!/bin/sh
# Write the API base URL from the container's environment into config.js,
# so one image works for any backend URL without rebuilding.
set -eu
escaped=$(printf '%s' "${API_BASE_URL:-}" | sed 's/\\/\\\\/g; s/"/\\"/g')
printf 'window.__APP_CONFIG__ = { API_BASE_URL: "%s" };\n' "$escaped" > /usr/share/nginx/html/config.js
echo "runtime config: API_BASE_URL=${API_BASE_URL:-<unset>}"
