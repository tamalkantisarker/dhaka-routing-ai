# Free hosting

The full Python app includes its OSM extract in `data/`. No API keys or local
Downloads folder are required. Use Python 3.12 and `app.py` as the entrypoint.

## Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io/> using the GitHub account that owns
   `tamalkantisarker/dhaka-routing-ai`.
2. Create an app from repository `tamalkantisarker/dhaka-routing-ai`, branch `main`,
   entrypoint `app.py`.
3. In Advanced settings choose Python 3.12. No secrets are needed.
4. Deploy, wait for the build, and open the resulting `*.streamlit.app` URL.

Deploy form:
<https://share.streamlit.io/deploy?repository=tamalkantisarker%2Fdhaka-routing-ai&branch=main&mainModule=app.py>

## Render (alternative)

The included `render.yaml` requests one **Free** Python web service, without a
database or paid disk. The `.python-version` file selects Python 3.12.

Open <https://render.com/deploy?repo=https://github.com/tamalkantisarker/dhaka-routing-ai>
and deploy the Blueprint after signing in. Check that the service plan is Free.
The service starts Streamlit on Render's assigned `$PORT`; its health endpoint
is `/_stcore/health`.

Free Render services sleep after 15 minutes without incoming traffic and can
take about a minute to wake. Their filesystem is temporary: graph caches rebuild
after restarts. Road edits are per browser session and are not permanently saved.

## Verify the live app

- Open the public URL from another browser/device.
- Confirm that the map loads and the graph reports 30,444 nodes / 73,443 edges.
- Run UCS and A*, then edit traffic or close a chosen road and replan.
- Check the before/after table and Remove/Reset behaviour.
- Confirm `GET /_stcore/health` returns `200` and `ok`.

A source repository or deploy-form link is not a deployed app. Hosting is complete
only after the provider returns a live URL and the app is checked there.
