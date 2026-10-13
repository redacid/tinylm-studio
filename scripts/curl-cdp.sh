# 1. Check that Chrome is up
curl -s http://127.0.0.1:9222/json/version

# 2. Open a tab with the page (newer Chrome requires PUT)
TAB=$(curl -s -X PUT "http://127.0.0.1:9222/json/new?https://store.epicgames.com/news/crimson-desert-secrets-you-might-miss?lang=ru")
ID=$(echo "$TAB" | jq -r .id)
sleep 3   # wait for JS to render

# 3. Get the rendered HTML
echo '{"id":1,"method":"Runtime.evaluate","params":{"expression":"document.documentElement.outerHTML","returnByValue":true}}' \
  | websocat -n1 "ws://127.0.0.1:9222/devtools/page/$ID" | jq -r .result.result.value

# 4. Close the tab
curl -s "http://127.0.0.1:9222/json/close/$ID"