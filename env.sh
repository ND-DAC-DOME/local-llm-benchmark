# Load the repo's settings into the current shell:  . ./env.sh
# Order matters: local overrides first (config.local.env, .env), then config.env, whose ${VAR:-default} lines
# keep anything already set and derive SSL_CERT_FILE / CURL_CA_BUNDLE from SPARK_CA_BUNDLE.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
[ -f "$_root/config.local.env" ] && . "$_root/config.local.env"
[ -f "$_root/.env" ] && . "$_root/.env"

# The Spark's private CA has to be added to the system CAs, not put in their place: SSL_CERT_FILE is global
# to the Python process, and bench.py also talks to huggingface.co for the datasets. Build a combined bundle
# next to the Spark certificate and point both variables at it (unless they are already set).
if [ -n "${SPARK_CA_BUNDLE:-}" ] && [ -f "$SPARK_CA_BUNDLE" ] && [ -z "${SSL_CERT_FILE:-}" ]; then
  for _sys in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
    [ -f "$_sys" ] || continue
    _combined="${SPARK_CA_BUNDLE%.*}-plus-system.crt"
    if [ ! -f "$_combined" ] || [ "$SPARK_CA_BUNDLE" -nt "$_combined" ] || [ "$_sys" -nt "$_combined" ]; then
      cat "$_sys" "$SPARK_CA_BUNDLE" > "$_combined"
    fi
    SSL_CERT_FILE="$_combined"
    CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$_combined}"
    break
  done
  unset _sys _combined
fi

. "$_root/config.env"
set +a
unset _root
