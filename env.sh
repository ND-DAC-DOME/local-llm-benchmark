# Load the repo's settings into the current shell:  . ./env.sh
# Order matters: local overrides first (config.local.env, .env), then config.env, whose ${VAR:-default} lines
# keep anything already set and derive SSL_CERT_FILE / CURL_CA_BUNDLE from SPARK_CA_BUNDLE.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
[ -f "$_root/config.local.env" ] && . "$_root/config.local.env"
[ -f "$_root/.env" ] && . "$_root/.env"
. "$_root/config.env"
set +a
unset _root
